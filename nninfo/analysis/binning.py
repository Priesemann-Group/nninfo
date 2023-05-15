import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from collections import Counter
from scipy.stats import norm
from sxpid import SxPID
from nninfo.config import CLUSTER_MODE

__all__ = ["Binning"]


class Binning(ABC):
    """
    Used for discretizing continuous activation data and produces a pmf
    from the discretized values.
    """

    def __init__(self, groupings, **kwargs):
        """
        Args:
            groupings:  List of PID variables, each defined as a list of neuron ids
        """
        self._params = kwargs
        self._groupings = groupings
        self._counter = None
        self.reset()

    def reset(self):
        self._counter = Counter()

    @classmethod
    def from_binning_method(
        cls, binning_method, groupings, source_id_lists, target_id_list, **kwargs
    ):
        if binning_method == "fixed_size":
            return FixedSizeBinning(groupings, **kwargs)
        elif binning_method == "goldfeld:fixed_size":
            return GoldfeldBinning(source_id_lists, target_id_list, **kwargs)
        elif binning_method == "none":
            return NoneBinnig(groupings, **kwargs)
        else:
            raise NotImplementedError

    def apply(self, activations):
        """
        Bins a batch of activations
        activations: {neuron_id -> [activations]}
        groupings: [[neuron_ids_source1/target], [neuron_ids_source2], ..., [neuron_ids_sourceN/target]]
        """
        grouped_activations = zip(
            *[
                zip(
                    *[
                        self.discretize(activations[neuron_id], neuron_id)
                        for neuron_id in grouping
                    ]
                )
                for grouping in self._groupings
            ]
        )
        self._counter.update(grouped_activations)

    def get_pdf(self):
        total = sum(self._counter.values())
        return {rlz: count / total for (rlz, count) in self._counter.items()}

    def get_params(self):
        return {**self._params, "binning_method": self.get_binning_method()}

    @abstractmethod
    def discretize(self, activations, neuron_id):
        raise NotImplementedError

    @abstractmethod
    def get_binning_method(self):
        raise NotImplementedError


class NoneBinnig(Binning):
    """
    Binning method that expects already discretized values and therefore skips the discretization step
    """
    def __init__(self, groupings, **kwargs):
        """
        Expected kwargs:
            limits: {neuron_id -> (lower_limit, upper_limit) or "binary"}
            n_bins: number of bins
        """
        super().__init__(groupings, **kwargs)

    def get_binning_method(self):
        return "none"

    def discretize(self, activations, neuron_id):
        return activations

class FixedSizeBinning(Binning):
    def __init__(self, groupings, **kwargs):
        """
        Expected kwargs:
            limits: {neuron_id -> (lower_limit, upper_limit) or "binary"}
            n_bins: number of bins
        """
        super().__init__(groupings, **kwargs)

    def get_binning_method(self):
        return "fixed_size"

    def discretize(self, activations, neuron_id):
        """

        Args:
            activations (numpy array): all activations
            neuron_id: id of the neuron on which discretization should be performed

        Returns: (numpy array) discretized version of the activations of neuron
            with neuron_id

        """
        limits = self._params["limits"][neuron_id]
        if limits == "binary":
            activations = activations.astype(np.int64)
            return activations
        elif isinstance(limits, tuple) and limits[0] != -np.inf and limits[1] != np.inf:
            binning_range = limits[1] - limits[0]
            bin_size = binning_range / self._params["n_bins"]
            discretized = np.floor(activations / bin_size).astype("int")
        else:
            raise NotImplementedError(
                "Applying discretization to layer with "
                + str(limits)
                + " limits is not possible"
            )
        return discretized


class GoldfeldBinning(FixedSizeBinning):
    def __init__(self, source_id_lists, target_id_list, **kwargs):
        super().__init__(None, **kwargs)
        """
        All sources are expected to be noisy, the target is assumed non-noisy.
        Expected kwargs:
            std_dev: standard deviation of gaussians
            limits: {neuron_id -> (lower_limit, upper_limit) or "binary"}
            n_bins: number of bins inside limits
            extra_bin_factor: Determines how many bins to add outside of limits as 2 * ceil(bin_width * std_dev / bin_size)
            prob_threshold: Events with probability mass lower than this threshold are excluded from PID for performance. pmf is normalized after exclusion.
        """

        self._source_id_lists = source_id_lists
        self._target_id_list = target_id_list

        beta = self._params["std_dev"]

        self._bin_centers = []

        extra_bin_factor = self._params["extra_bin_factor"]

        # Calculate center positions of bins of relevant neurons (i.e. those which appear in a grouping)
        for grouping in source_id_lists:
            for neuron_id in grouping:

                limits = self._params["limits"][neuron_id]
                binning_range = limits[1] - limits[0]
                bin_size = binning_range / self._params["n_bins"]
                n_extra_bins = int(np.ceil(extra_bin_factor * beta / bin_size))

                neuron_bin_centers = (
                    np.linspace(
                        limits[0] - n_extra_bins * bin_size,
                        limits[1] + n_extra_bins * bin_size,
                        self._params["n_bins"] + 2 * n_extra_bins,
                        endpoint=False,
                    )
                    + bin_size / 2
                )

                self._bin_centers.append(neuron_bin_centers)

        # Dictionary from target tuples to full ndarray source pmfs.
        self._probability_dict = {}

        # Full unnormalized pmf for all bin combinations
        self._accumulated_probabilites = np.zeros(
            shape=tuple(len(nbc) for nbc in self._bin_centers)
        )

        # Initialize normal distribution
        self._normal = norm(loc=0, scale=beta).pdf

    def get_binning_method(self):
        return "goldfeld:fixed_size"

    def apply(self, activations):
        """
        Apply Goldfeld binnning to a batch of activations, i.e. evaluate multidimensional Gaussians 
        with the distance of the sample from the bin centers.
        """

        # First digitize the non-noisy target varible
        digitized_target = list(
            zip(
                *[
                    self.discretize(activations[neuron_id], neuron_id)
                    for neuron_id in self._target_id_list
                ]
            )
        )

        # Find relevant activations source, i.e. from neurons of interest for the requested PID groupings
        relevant_activations = np.hstack(
            tuple(
                activations[neuron_id][:, np.newaxis]
                for grouping in self._source_id_lists
                for neuron_id in grouping
            )
        )

        # For each sample, calculate the contributions to the accumulated_probabilities by
        # evaluating 1D Gaussians for the dimensions separately and (outer) multiplying them.
        # This exploits the fact that our evaluation points are on a cartesian grid and the Gaussian
        # Factorizes over dimensions.
        for i, sample in tqdm(
            enumerate(relevant_activations),
            total=len(relevant_activations),
            disable=CLUSTER_MODE,
        ):  # vectorize!
            sample_contributions = 1
            for neuron_sample, neuron_bin_centers in zip(sample, self._bin_centers):
                neuron_marginal = self._normal(neuron_bin_centers - neuron_sample)
                sample_contributions = np.multiply.outer(
                    sample_contributions, neuron_marginal
                )

            if digitized_target[i] in self._probability_dict:
                self._probability_dict[digitized_target[i]] += sample_contributions
            else:
                self._probability_dict[digitized_target[i]] = sample_contributions

    def get_pdf(self):
        """
        Applies the threshold, groups dimensions of _accumulated_probabilities according to groupings,
        reencodes the accumulated_probabilities in a sparse matrix format (COO)
        and normalizes it.
        """

        # Find that a full pmf array would have after applying the variable grouping for PID analysis
        grouped_shape = ()
        i = 0
        for grouping in self._source_id_lists:
            group_bins = 1
            for _ in grouping:
                group_bins *= len(self._bin_centers[i])
                i += 1
            grouped_shape += (group_bins,)
        grouped_shape += (len(self._probability_dict),)

        # Create coordinate array according to shape and flatten it
        coords = (
            np.indices(grouped_shape).reshape(len(grouped_shape), -1).T
        )  # shape=(n_samples, n_PID_variables)

        flattened_pmf = np.stack(
            list(self._probability_dict.values()), axis=-1
        ).flatten()  # shape=n_samples

        # normalize
        flattened_pmf = flattened_pmf / flattened_pmf.sum()

        # Apply threshold or find it first
        if "prob_threshold" in self._params:
            eps = self._params["prob_threshold"]
        else:
            q = self._params[
                "min_incl_prob"
            ]  # Minimum total probability to be included in the PID
            eps = self.find_thr(flattened_pmf, q)

        above_threshold_indices = np.nonzero(flattened_pmf >= eps)

        filtered_coords = coords[above_threshold_indices]
        filtered_pmf = flattened_pmf[above_threshold_indices]

        print(filtered_pmf.sum())
        print(filtered_pmf.sum() - filtered_pmf.min())

        # Normalize
        normalized_pmf = filtered_pmf / np.sum(filtered_pmf)
        print("Remaining total pmf:", np.sum(filtered_pmf) / np.sum(flattened_pmf))

        # Create SxPID pdf directly, without creating dictionary first. Make coords F-contiguous for performance reasons.
        return SxPID.PDF(np.asfortranarray(filtered_coords), normalized_pmf)

    def find_thr(self, a, q, bias_l=0, bias_g=0, first_call=True):
        """
        Args:
            a: normalized array of weights
            q: quantile
        Returns:
            thr: largest threshold such that a[a >= thr].sum() >= q
        """
        if len(a) == 1:
            return a[0]

        if len(a) == 2:
            min = np.min(a)
            max = np.max(a)
            return max if max + bias_g >= q else min

        pivot_idx = int(q * len(a)) if first_call else int(len(a) / 2)
        partitioned = np.partition(a, kth=pivot_idx, axis=None)

        sum_l = np.sum(partitioned[:pivot_idx]) + bias_l
        sum_g = np.sum(partitioned[pivot_idx + 1 :]) + bias_g

        if sum_l < 1 - q and sum_g < q:
            return partitioned[pivot_idx]

        if sum_g > q:
            return self.find_thr(
                partitioned[pivot_idx:],
                q,
                bias_l=sum_l,
                bias_g=bias_g,
                first_call=False,
            )
        else:
            return self.find_thr(
                partitioned[: pivot_idx + 1],
                q,
                bias_l=bias_l,
                bias_g=sum_g,
                first_call=False,
            )
