import copy

import pandas as pd
from sxpid import SxPID
from broja2pid import BROJA_2PID

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

        self.T = sorted(target_id_list)
        self.S = [sorted(s) for s in source_id_lists]

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

    def _perform_sxpid(self, sxpid_pdf):
        results = SxPID.pid(
            sxpid_pdf, verbose=0 if CLUSTER_MODE else 2, n_threads=N_WORKERS, pointwise=False
        )

        # Convert results to a row in a dataframe. The multiindex column names are ("informative"/"misinformative"/"average", str(antichain))
        inf = pd.DataFrame({str(k): [v[0]] for k, v in results.items()})
        misinf = pd.DataFrame({str(k): [v[1]] for k, v in results.items()})
        avg = pd.DataFrame({str(k): [v[2]] for k, v in results.items()})

        return pd.concat([inf, misinf, avg], axis=1, keys=['inf_pid', 'misinf_pid', 'avg_pid'])

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
                     'reing': _perform_reing}
