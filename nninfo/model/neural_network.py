import copy
from dataclasses import dataclass, field
from functools import cache
from typing import Optional, Union, Tuple, List
from ast import literal_eval

import torch.nn as nn
import torch
import numpy as np
import scipy as scp
import yaml

import nninfo
from ..file_io import NoAliasDumper
from .quantization import quantizer_list_factory

Limits = Union[Tuple[float, float], str]

class RandomRotation(nn.Module):

    def __init__(self, dim, rng_seed):
        super(RandomRotation, self).__init__()
        self.dim = dim
        self.rng_seed = rng_seed

        rng = np.random.default_rng(self.rng_seed)
        self.rotation_matrix = torch.tensor(scp.stats.special_ortho_group.rvs(dim, random_state=rng), dtype=torch.float32)

    def forward(self, x):
        return torch.matmul(x, self.rotation_matrix)

CONNECTION_LAYERS_PYTORCH = {
    "input": lambda: (lambda x: x),
    "identity": lambda: (lambda x: x),
    "linear": nn.Linear,
    "dropout": nn.Dropout,
    "maxpool2d": nn.MaxPool2d,
    "conv2d": nn.Conv2d,
    "flatten": nn.Flatten,
    "random_rotation": RandomRotation,
}

ACTIV_FUNCS_PYTORCH = {
    None: lambda: (lambda x: x),
    "input": lambda: (lambda x:x),  
    "relu": lambda: torch.relu,
    "tanh": lambda: torch.tanh,
    "hardtanh": lambda: nn.Hardtanh(),
    "sigmoid": lambda: torch.sigmoid,
    "softmax": lambda: torch.nn.functional.softmax,
    "log_softmax": lambda: torch.nn.functional.log_softmax,
    "softmax_output": lambda: (lambda x: x),
}

ACTIV_FUNCS_BINNING_LIMITS = {
    None: (-np.inf, np.inf),
    "input": None,
    "relu": (0.0, np.inf),
    "tanh": (-1.0, 1.0),
    "hardtanh": (-1.0, 1.0),
    "sigmoid": (0.0, 1.0),
    "softmax": (0.0, 1.0),
    "log_softmax": (-np.inf, 0.0),
    "softmax_output": (0.0, 1.0),
}

INITIALIZERS_PYTORCH = {
    "xavier": nn.init.xavier_uniform_,
    "he_kaiming": nn.init.kaiming_uniform_,
    "he_kaiming_normal": nn.init.kaiming_normal_,
}


@dataclass
class LayerInfo:
    """A layer of a neural network.
    
        Consists of a connection layer and an activation function.
    """
    
    connection_layer: str
    activation_function: str
    connection_layer_kwargs: Optional[dict] = field(default_factory=dict)
    activation_function_kwargs: Optional[dict] = field(default_factory=dict)

    @staticmethod
    def from_config(layer_dict):
        return LayerInfo(
            connection_layer=layer_dict["connection_layer"],
            connection_layer_kwargs=layer_dict["connection_layer_kwargs"],
            activation_function=layer_dict["activation_function"],
            activation_function_kwargs=layer_dict["activation_function_kwargs"],
        )
    
    def to_config(self):
        return {
            "connection_layer": self.connection_layer,
            "connection_layer_kwargs": self.connection_layer_kwargs,
            "activation_function": self.activation_function,
            "activation_function_kwargs": self.activation_function_kwargs,
        }
    
@dataclass(frozen=True)
class NeuronID():
    """Index of a neuron in a neural network.

    Consists of a layer label and a neuron index.
    The input layer is labeled "X", the output layer "Y",
    and the hidden layers "L1", "L2", ...
    """
    layer: str
    index: Union[int, Tuple[int, ...]]

    @staticmethod
    def to_yaml(dumper, data):
        # Dumps a NeuronID to a string in YAML.
        return dumper.represent_scalar("!NeuronID", f"layer={data.layer}, index={data.index}")
        
    @staticmethod
    def from_yaml(loader, node):
        # Loads a NeuronID from a string in YAML.
        value = loader.construct_scalar(node)
        layer=value.split("layer=")[1].split(",")[0]
        index=value.split("index=")[1]
        index=literal_eval(index)
        return NeuronID(layer, index)
    
NoAliasDumper.add_representer(NeuronID, NeuronID.to_yaml)
yaml.SafeLoader.add_constructor("!NeuronID", NeuronID.from_yaml)

class NeuralNetwork(nninfo.exp_comp.ExperimentComponent, nn.Module):
    """
    Model that is trained and analysed.

    CUDA acceleration is not implemented yet, but will certainly be possible in the future.
    """

    def __init__(
            self,
            layer_infos: List[LayerInfo],
            init_str,
            **kwargs
    ):
        """
        Creates a new instance of NeuralNetwork and sets all structural parameters of the model.

        Important comment: the external indexing of the layers is 1,...,n for convenience.
        However, I could not find a way to use this indexing also for the inner structure. Maybe
        we might change that in the future to avoid errors.

        Args:

        Keyword Args:
            noise_stddev (float): in case of a noisy neural network
        """
        # call pytorch super class
        super().__init__()

        self._layer_infos = layer_infos
        self._params = kwargs

        self.n_layers = len(layer_infos)

        self._connection_layers = []
        self._activ_funcs = []
        self._module_list = nn.ModuleList()

        for layer_info in layer_infos:

            try:
                activation_function_factory = ACTIV_FUNCS_PYTORCH[layer_info.activation_function]
            except KeyError:
                raise ValueError(f"Activation function {layer_info.activation_function} not supported.")
            
            activation_function = activation_function_factory(**layer_info.activation_function_kwargs)

            self._activ_funcs.append(activation_function)

            try:
                connection_layer_factory = CONNECTION_LAYERS_PYTORCH[layer_info.connection_layer]
            except KeyError:
                raise ValueError(f"Connection layer {layer_info.connection_layer} not supported.")
            
            connection_layer = connection_layer_factory(**layer_info.connection_layer_kwargs)

            self._connection_layers.append(connection_layer)
            if isinstance(connection_layer, nn.Module):
                self._module_list.append(connection_layer)

        self._initializer = None
        self._init_str = init_str

        self.init_weights()

    @staticmethod
    def from_config(config):
        """
        Creates a NeuralNetwork instance from a config dictionary.

        Args:
            config (dict): Dictionary with all the necessary information to create a NeuralNetwork instance.

        Returns:
            NeuralNetwork: Instance of NeuralNetwork with the given configuration.
        """

        config = config.copy()
        net_type = config.pop("net_type")


        layer_configs = [LayerInfo.from_config(layer_info) for layer_info in config.pop("layers")]

        if net_type == "feedforward":
            return NeuralNetwork(layer_configs, **config)
        elif net_type == "noisy_feedforward":
            return NoisyNeuralNetwork(layer_configs, **config)

        raise KeyError(f"Unknown network type '{net_type}'")

    def to_config(self):
        """
        Is called when the experiment saves all its components at the beginning of an experiment.

        Returns:
            dict: Dictionary with all the network settings necessary to create the network again
                for rerunning and later investigation.
        """
        param_dict = {
            "layers": [layer_info.to_config() for layer_info in self._layer_infos],
            "net_type": "feedforward",
            "init_str": self._init_str,
        }

        return copy.deepcopy(param_dict)
    
    @cache
    def get_layer_sizes(self):
        """
        Returns:
            list of int: List of layer output sizes.
        """
        shapes = [self.parent.task.x_dim]

        for i in range(1, self.n_layers):

            layer_info = self._layer_infos[i]

            if "out_features" in layer_info.connection_layer_kwargs:
                shapes.append(layer_info.connection_layer_kwargs["out_features"])
            elif "out_channels" in layer_info.connection_layer_kwargs:
                shapes.append(layer_info.connection_layer_kwargs["out_channels"])
            else:
                shapes.append(None)

    def get_layer_structure_dict(self):
        layer_dict = dict()
        for i, size in enumerate(self.get_layer_sizes()):
            if i == 0:
                layer_label = "X"
            else:
                layer_label = "L" + str(i)
            neuron_id_list = []
            for n in range(size):
                neuron_id_list.append(NeuronID(layer_label, n + 1))
            layer_dict[layer_label] = neuron_id_list

        layer_label = "Y"
        neuron_id_list = []
        for n in range(self.get_layer_sizes()[-1]):
            neuron_id_list.append(NeuronID(layer_label, n + 1))
        layer_dict[layer_label] = neuron_id_list
        return layer_dict

    def get_binning_limits(self):
        """
        returns: {neuron_id -> (low, high) or "binary"}
        """
        structure = self.get_layer_structure_dict()
        limit_dict = dict()
        for layer_label in structure:
            for neuron_id in structure[layer_label]:
                if layer_label == "X" or layer_label == "Y":
                    limit_dict[neuron_id] = self.parent.task.get_binning_limits(
                        layer_label
                    )
                elif layer_label.startswith("L"):
                    i = int(layer_label.lstrip("L"))
                    limit_dict[neuron_id] = ACTIV_FUNCS_BINNING_LIMITS[
                        self._layer_infos[i].activation_function
                    ]
                else:
                    raise NotImplementedError(
                        "Value " + layer_label + "not allowed for layer_label."
                    )
        return limit_dict

    def get_limits_list(self) -> List[Limits]:
        """
        Currently returns None for the input Limits
        returns: [(low, high) or "binary"]
        """
        return [ACTIV_FUNCS_BINNING_LIMITS[layer_info.activation_function] for layer_info in self._layer_infos]

    def forward(self, x, quantizers, return_activations=False, apply_output_softmax=None):
        """

        Forward pass for the network, given a batch.

        Args:
            x (torch tensor): batch from dataset to be fed into the network
        Returns:
            torch tensor: output of the network
        """

        if apply_output_softmax is None:
            apply_output_softmax = return_activations

        if return_activations:
            activations = {}

        for i in range(self.n_layers):

            x = self._connection_layers[i](x)
            x = self._activ_funcs[i](x)

            if apply_output_softmax and self._layer_infos[i].activation_function == 'softmax_output':
                x = torch.softmax(x, dim=1)

            x = quantizers[i](x)

            if return_activations:
                activations["L" + str(i + 1)] = x.detach().numpy()

        if return_activations:
            return x, activations
        
        return x

    def extract_activations(self, x, quantizer_params):
        """
        Extracts the activities using the input given. Used by Analysis. Outputs
        activities together in a dictionary (because of variable sizes of the layers).

        Args:
            x (torch tensor): batch from dataset to calculate the activities on. Typically
                feed the entire dataset in one large batch.
            before_noise (bool): In a noisyNN, sample before or after applying noise

        Returns:
            dict: dictionary with each neuron_id as key,
                labeled "L1",..,"L<n>",
                where "L1" corresponds to the output of the first layer and "L<n>" to
                the output of the final layer. Notice that this indexing is different
                from the internal layer structures indices (they are
                uncomfortable to change).
        """
        test_quantizers = quantizer_list_factory(
            quantizer_params, self.get_limits_list())

        with torch.no_grad():
            _, activations = self.forward(x, quantizers=test_quantizers, return_activations=True)

        return activations

    def init_weights(self, randomize_seed=False):
        """
        Initialize the weights using the init_str that was set at initialization
        (one of the initializers provided in INITIALIZERS_PYTORCH).

        Args:
            randomize_seed (bool): If true, the torch seed is reset before the initialization.
        """

        self._initializer = INITIALIZERS_PYTORCH[self._init_str]
        if randomize_seed:
            torch.seed()
        self.apply(self._init_weight_fct)

    def _init_weight_fct(self, m):
        if isinstance(m, nn.Linear):
            if self._init_str == "xavier":
                self._initializer(m.weight, gain=5./3.)
            else:
                self._initializer(m.weight)
        elif isinstance(m, nn.Conv2d):
            if self._init_str == "xavier":
                self._initializer(m.weight, gain=np.sqrt(2))
            else:
                self._initializer(m.weight)
        elif isinstance(m, nn.Sequential):
            if len(m) == 2 and isinstance(m[0], nn.Linear):
                if self._init_str == "xavier":
                    self._initializer(m[0].weight, gain=5./3.)
                else:
                    self._initializer(m[0].weight)
        elif not isinstance(m, (nn.ModuleList, NeuralNetwork, nn.MaxPool2d, nn.Flatten)):
            raise NotImplementedError(
                f"Weight initialization for {m} is not implemented."
            )


    def __call__(self, x, quantizers):
        return self.forward(x, quantizers)

    def get_input_output_dimensions(self):
        input_dim = self.get_layer_sizes()[0]
        output_dim = self.get_layer_sizes()[-1]
        return input_dim, output_dim

    def neuron_ids(self, only_real_neurons=False):
        """
        Create a simple list of all nodes of the network
        (including input, target, bias nodes).

        Args:
            only_real_neurons: Whether you want to only include neurons
                whose ids begin with 'L'. Default is False

        Returns:
            list: neuron ids
        """
        names = []
        if not only_real_neurons:
            names.append(("B", (1,)))
        for layer_name, neurons in self.get_layer_structure_dict().items():
            if (not only_real_neurons) or (
                only_real_neurons and layer_name.startswith("L")
            ):
                for neuron in neurons:
                    names.append(neuron)
        return names

    def connectome(self):
        """
        Returns:
            an empty connectome matrix
        """
        neuron_list = self.neuron_ids()
        connectome = [[]]
        connectome[0].append("input_neuron_ids")
        connectome[0].extend(neuron_list)
        for neuron in neuron_list:
            connectome.append([neuron])
            connectome[-1].extend([float(-1) for i in range(len(neuron_list))])
        return connectome

    def trainable_parameters(self):
        """
        Create a list of all trainable parameters.
        Ugly code still.

        Returns:
            list: List of all trainable parameters.
                dim 0: parameters
                dim 1: input neuron_id, output neuron_id, parameter_id
        """
        connectome = self.connectome()
        param_list = []
        param_index = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                _, internal_index, wb = name.split(".")
                layer_index = int(internal_index) + 1
                if wb == "weight":
                    for i in range(param.shape[0]):
                        for j in range(param.shape[1]):
                            input_layer, output_layer = (
                                "L" + str(layer_index - 1),
                                "L" + str(layer_index),
                            )
                            if input_layer == "L0":
                                input_layer = "X"
                            k = connectome[0].index((input_layer, (j + 1,)))
                            l = connectome[0].index((output_layer, (i + 1,)))
                            connectome[k][l] = param_index
                            param_list.append(
                                [connectome[0][k], connectome[0][l], param_index]
                            )
                            param_index += 1
                elif wb == "bias":
                    for i in range(param.shape[0]):
                        k = connectome[0].index(("B", (1,)))
                        l = connectome[0].index(
                            ("L" + str(layer_index), (i + 1,)))
                        connectome[k][l] = param_index
                        param_list.append(
                            [connectome[0][k], connectome[0][l], param_index]
                        )
                        param_index += 1
        return param_list


class NoisyNeuralNetwork(NeuralNetwork):
    def forward(self, x, return_activations=False, save_before_noise=False, quantizer=None, apply_output_softmax=False):
        
        if return_activations:
            activations = {}

        for i in range(self.n_layers):
            x = self.layers[i](x)
            x = self._activ_funcs[i](x)

            if apply_output_softmax and self._activ_func_str[i] == 'softmax_output':
                x = torch.softmax(x, dim=1)

            # add gaussian noise to the layer with stddev noise_stddev
            if return_activations and save_before_noise:
                activations["L" + str(i + 1)] = x.detach().numpy()

            # if i != self.n_layers - 1:
            limits = ACTIV_FUNCS_BINNING_LIMITS[self._activ_func_str[i]]
            sampled_noise = torch.empty(x.shape).normal_(
                mean=0, std=self._params["noise_stddev"] * (limits[1]-limits[0])
            )
            x = x + sampled_noise

            if return_activations and not save_before_noise:
                activations["L" + str(i + 1)] = x.detach().numpy()

        if return_activations:
            return x, activations

        return x

    def to_config(self):
        param_dict = super.to_config()
        param_dict["noise_stddev"] = self._params["noise_stddev"]
        return param_dict
