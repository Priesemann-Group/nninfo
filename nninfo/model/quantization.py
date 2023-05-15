from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Union, Tuple, List

import numpy as np
import torch

Limits = Union[Tuple[float, float], str]

@dataclass
class Quantizer(ABC):
    """
        n_levels: Number of equidistant quantization levels
    """

    n_levels: int
    limits: Limits

    @abstractmethod
    def __call__(self, x: torch.Tensor):
        pass


@dataclass
class StochasticQuantizer(Quantizer):
    """Stochastically rounds either up or down to the nearest Q3 quantization level.
    The probability of rounding up is proportional to the normalized remainder.
    """

    def __post_init__(self):

        # Make sure the limits are finite
        assert np.isfinite(self.limits[0]) and np.isfinite(
            self.limits[1]), f"Quantization limits are not finite {self.limits}"

        self.scale = (self.limits[1] - self.limits[0]) / (self.n_levels - 1)

    def __call__(self, x: torch.Tensor):


        # Make sure all activations are within the limits
        assert torch.all(x >= self.limits[0]) and torch.all(
            x <= self.limits[1]), f"Activations are not within the limits {self.limits}"

        x_normalized = (x - self.limits[0]) / self.scale

        ceil_prob = torch.remainder(x_normalized, 1)
        ceil_mask = torch.bernoulli(ceil_prob)

        floored = FloorNoGradient.apply(
            x_normalized) * self.scale + self.limits[0]
        ceiled = CeilNoGradient.apply(
            x_normalized) * self.scale + self.limits[0]

        return torch.where(ceil_mask == 1, ceiled, floored)

@dataclass
class Q3Quantizer(Quantizer):
    """The two outmost bins are only half the width than the others and round to the binning limits."""

    def __post_init__(self):

        # Make sure the limits are finite
        assert np.isfinite(self.limits[0]) and np.isfinite(
            self.limits[1]), f"Quantization limits are not finite {self.limits}"

        self.scale = (self.limits[1] - self.limits[0]) / (self.n_levels - 1)

    def __call__(self, x: torch.Tensor):

        # Make sure all activations are within the limits
        assert torch.all(x >= self.limits[0]) and torch.all(
            x <= self.limits[1]), f"Activations are not within the limits {self.limits}"

        return torch.round((x - self.limits[0]) / self.scale) * self.scale + self.limits[0]


class CeilNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.ceil()

    @staticmethod
    def backward(ctx, g):
        return g


class FloorNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.floor()

    @staticmethod
    def backward(ctx, g):
        return g


def quantizer_factory(rounding_point: str, levels: int, limits: Limits) -> Quantizer:
    if rounding_point == "stochastic":
        return StochasticQuantizer(levels, limits)
    elif rounding_point == "center_saturating":
        return Q3Quantizer(levels, limits)
    else:
        raise NotImplementedError(
            f'No quantizer with rounding point {rounding_point} is available.')


def quantizer_list_factory(quantizer_params: Union[None, dict, list], limits: List[Limits]) -> List[Quantizer]:

    if quantizer_params is None:
        return [lambda x: x] * len(limits)

    if isinstance(quantizer_params, dict):
        quantizer_param_list = [quantizer_params] * len(limits)
    elif isinstance(quantizer_params, list):
        quantizer_param_list = quantizer_params

    return [(quantizer_factory(**params, limits=limits) if params is not None else (lambda x: x)) for params, limits in zip(quantizer_param_list, limits)]

