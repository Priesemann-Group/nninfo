import copy
import pickle

import pandas as pd
import numpy as np
from sxpid import SxPID
from broja2pid import BROJA_2PID
from tqdm import tqdm

from .measurement import Measurement
from .binning import Binning
from ..config import N_WORKERS, CLUSTER_MODE
from ..experiment import Experiment
from ..model.neural_network import NeuronID
from .. import logger
from .reing import compute_C_k_diffs

log = logger.get_logger(__name__)


class PIDMeasurement(Measurement):
    measurement_type = "pid"
    """
    A single PID Measurement, that can be saved into the file structure after computation.
    """

    def __init__(self,
                 experiment: Experiment,
                 measurement_id: str,
                 dataset_name: str,
                 pid_definition: str,
                 target_id_list: list[NeuronID],
                 source_id_lists: list[list[NeuronID]],
                 binning_kwargs,
                 pid_kwargs: dict = None,
                 dataset_kwargs: dict = None,
                 quantizer_params=None,
                 _load=False,
                 ):

        self._pid_definition = pid_definition
        self._pid_kwargs = pid_kwargs or {}

        self.T = target_id_list
        self.S = [s for s in source_id_lists]
        self.T = target_id_list
        self.S = source_id_lists

        self._n_sources = len(source_id_lists)

        self._binning_kwargs = binning_kwargs or {}

        super().__init__(
            experiment=experiment,
            measurement_id=measurement_id,
            dataset_name=dataset_name,
            dataset_kwargs=dataset_kwargs,
            quantizer_params=quantizer_params,
            _load=_load
        )

        binning_kwargs_copy = copy.deepcopy(binning_kwargs)
        binning_method = binning_kwargs_copy.pop("binning_method")
        self._binning = Binning.from_binning_method(
            binning_method,
            source_id_lists + [target_id_list],
            source_id_lists,
            target_id_list,
            **binning_kwargs_copy
        )

    def to_config(self):
        config = super().to_config()
        config["pid_definition"] = self._pid_definition
        config["pid_kwargs"] = self._pid_kwargs
        config["binning_kwargs"] = self._binning_kwargs
        config["target_id_list"] = self.T
        config["source_id_lists"] = self.S
        return config

    def _measure(self, run_id, chapter_id):

        # get the activations for the current run_id and chapter_id
        activations_iter = self._experiment.capture_activations(
            dataset=self._dataset_name,
            run_id=run_id,
            chapter_id=chapter_id,
            repeat_dataset=self._dataset_kwargs.get("repeat_dataset", 1),
            before_noise=self._binning_kwargs.get("before_noise", False),
            quantizer_params=self._quantizer_params,
        )

        self._binning.reset()
        for activations in activations_iter:
            self._binning.apply(activations)

        pdf_dict = self._binning.get_pdf()

        try:
            pid_function = self._pid_measures[self._pid_definition]
        except KeyError:
            raise NotImplementedError(
                f"PID definition {self._pid_definition} is not implemented."
            )

        results = pid_function(self, pdf_dict, **self._pid_kwargs)

        return results
    
    def _perform_sxpid(self, sxpid_pdf, pointwise=False):
        print(f"Performing sxpid with pointwise={pointwise}")
        if pointwise == 'target':
            return self._perform_sxpid_targetpointwise(sxpid_pdf)
        elif pointwise == False:
            return self._perform_sxpid_average(sxpid_pdf)
        else:
            raise NotImplementedError(f"Pointwise={pointwise} not implemented")
    
    def _perform_sxpid_average(self, sxpid_pdf):
        results = SxPID.pid(
            sxpid_pdf, verbose=0 if CLUSTER_MODE else 2, n_threads=N_WORKERS, pointwise=False
        )

        # Convert results to a row in a dataframe. The multiindex column names are ("informative"/"misinformative"/"average", str(antichain))
        inf = pd.DataFrame({str(k): [v[0]] for k, v in results.items()})
        misinf = pd.DataFrame({str(k): [v[1]] for k, v in results.items()})
        avg = pd.DataFrame({str(k): [v[2]] for k, v in results.items()})

        return pd.concat([inf, misinf, avg], axis=1, keys=['inf_pid', 'misinf_pid', 'avg_pid'])
    
    def _perform_sxpid_targetpointwise(self, sxpid_pdf):
        targetpointwise_dict, _ = SxPID.pid(
            sxpid_pdf, verbose=0, n_threads=N_WORKERS, pointwise="target", showProgress="print"
        )

        # targetpointwise_dict has the form {target: {antichain: (inf, misinf, avg)}}
        
        df = pd.DataFrame()

        # Iterate over all targets once
        for target in targetpointwise_dict.keys():

            # Convert the targetlocaldict to a single-row dataframe
            inf = pd.DataFrame({str(k): [v[0]] for k, v in targetpointwise_dict[target].items()})
            misinf = pd.DataFrame({str(k): [v[1]] for k, v in targetpointwise_dict[target].items()})
            avg = pd.DataFrame({str(k): [v[2]] for k, v in targetpointwise_dict[target].items()})

            # Combine the three dataframes into a single dataframe
            targetlocal_df = pd.concat([inf, misinf, avg], axis=1, keys=['inf_pid', 'misinf_pid', 'avg_pid'])

            # Add the target ID as a column
            targetlocal_df['target'] = [target]

            # Append the targetlocal_df to the main dataframe
            df = pd.concat([df, targetlocal_df], ignore_index=True)

        return df
    
    def _perform_deg_red(self, pdf_dict, entropy_only=False):

        if entropy_only:
            return self._perform_deg_red_entropy(pdf_dict)

        print("Computing degree of redundancy")
        
        # Convert to SxPID.PDF
        pdf = SxPID.PDF.from_dict(pdf_dict)
        print("Converted to SxPID.PDF")

        # Compute total MI
        p_ST = pdf.probs
        p_S = self.marginalize_same_shape(pdf, self._n_sources)
        p_T = self.marginalize_same_shape(pdf, *range(self._n_sources))

        total_MI = self._compute_MI(p_ST, p_ST, p_S, p_T)
        #print(f"Total MI: {total_MI}")

        # Compute the MI of each source with the target
        source_MI = []
        minus_source_MI = []
        for i in range(self._n_sources):
            print(f"{i}", end=" ", flush=True)
            p_SiT = self.marginalize_same_shape(pdf, *(tuple(range(i)) + tuple(range(i+1, self._n_sources))))
            p_Si = self.marginalize_same_shape(pdf, *(tuple(range(i)) + tuple(range(i+1, self._n_sources+1))))
            source_MI.append(self._compute_MI(p_ST, p_SiT, p_Si, p_T))
            #print(f"Source {i} MI: {source_MI[-1]}")

            # Compute the MI of each source with the target, conditioned on the other sources
            p_SiminusT = self.marginalize_same_shape(pdf, i)
            p_Siminus = self.marginalize_same_shape(pdf, *(i,self._n_sources))
            minus_source_MI.append(self._compute_MI(p_ST, p_SiminusT, p_Siminus, p_T))
            #print(f"Source {i} cond MI: {source_cond_MI[-1]}")
        print()

        # Store the results in a dataframe
        results = pd.DataFrame({
            'total_MI': [total_MI],
            **{f'source_MI{i}': [source_MI] for i, source_MI in enumerate(source_MI)},
            **{f'allbut_source_MI{i}': [source_cond_MI] for i, source_cond_MI in enumerate(minus_source_MI)}
        })

        return results
    
    def _perform_deg_red_entropy(self, pdf_dict):

        print("Computing degree of redundancy for entropy only")
        
        # Convert to SxPID.PDF
        pdf = SxPID.PDF.from_dict(pdf_dict)
        print("Converted to SxPID.PDF")

        # Compute total entropy
        p_S = pdf.probs

        total_H = self._compute_H(p_S, p_S)
        #print(f"Total MI: {total_MI}")

        # Compute the MI of each source with the target
        source_H = []
        allbut_source_H = []

        from multiprocessing import Pool
        with Pool(32) as p:
            results = p.starmap(PIDMeasurement._perform_deg_red_entropy_single_source, [(pdf, i) for i in range(self._n_sources)])
        source_H, allbut_source_H = zip(*results)

        """
        for i in range(self._n_sources):
            print(f"{i}", end=" ", flush=True)
            p_Si = self.marginalize_same_shape(pdf, *(tuple(range(i)) + tuple(range(i+1, self._n_sources))))
            source_H.append(self._compute_H(p_S, p_Si))
            #print(f"Source {i} MI: {source_MI[-1]}")

            # Compute the entropy of all sources except the current source
            p_allbut_S= self.marginalize_same_shape(pdf, i)
            allbut_source_H.append(self._compute_H(p_S, p_allbut_S))
            #print(f"Source {i} cond MI: {source_cond_MI[-1]}")
        print()
        """

        # Store the results in a dataframe
        results = pd.DataFrame({
            'total_H': [total_H],
            **{f'source_H{i}': [_source_H] for i, _source_H in enumerate(source_H)},
            **{f'allbut_source_H{i}': [_allbut_source_H] for i, _allbut_source_H in enumerate(allbut_source_H)}
        })

        return results
    
    @staticmethod
    def _perform_deg_red_entropy_single_source(pdf, source_idx):
        print(source_idx)
        n_sources = pdf.coords.shape[1]
        p_S = pdf.probs
        p_Si = PIDMeasurement.marginalize_same_shape(pdf, *(tuple(range(source_idx)) + tuple(range(source_idx+1, n_sources))))
        H_Si = PIDMeasurement._compute_H(p_S, p_Si)

        p_allbut_S= PIDMeasurement.marginalize_same_shape(pdf, source_idx)
        H_allbut_S = PIDMeasurement._compute_H(p_S, p_allbut_S)

        return H_Si, H_allbut_S

    @staticmethod
    def marginalize_same_shape(pdf, *coords):
        """
        Args:
            coord: Coordinate to marginalize out.

        Returns:
            Probability masses of the marginalized PDF in the same shape as the input PDF.
        """
        coords = [
            coord if coord >= 0 else pdf.coords.shape[1] + coord for coord in coords
        ]

        coords_reduced = np.delete(pdf.coords, obj=coords, axis=1)

        unique, inverse = np.unique(
            coords_reduced, return_index=False, return_inverse=True, axis=0
        )

        newprobs = np.bincount(inverse, weights=pdf.probs)

        return newprobs[inverse]
        
    def _compute_MI(self, weight, p_ST, p_S, p_T):
        return np.sum(weight * np.log2(p_ST / (p_S * p_T)))
    
    @staticmethod
    def _compute_H(weight, p_S):
        return np.sum(weight * np.log2(1 / p_S))

    
    def _perform_brojapid(self, pdf_dict):
        

        # BROJA-PID expects the target to be the first variable:
        broja_pdf_dict = {(key[-1],) + key[:-1]
                           : value for key, value in pdf_dict.items()}

        results = BROJA_2PID.pid(
            broja_pdf_dict, cone_solver="ECOS"
        )

        avg_dict = {
            '((1, 2,),)': [results["CI"]],
            '((1,),)': [results["UIY"]],
            '((2,),)': [results["UIZ"]],
            '((1,), (2,))': [results["SI"]]
        }

        return pd.DataFrame(avg_dict)

    def _perform_reing(self, reing_pdf):
        results_dict = compute_C_k_diffs(reing_pdf)

        return pd.DataFrame({str(k): [v] for k, v in results_dict.items()})

    _pid_measures = {'sxpid': _perform_sxpid,
                     'brojapid': _perform_brojapid,
                     'deg_red': _perform_deg_red,
                     'reing': _perform_reing}
