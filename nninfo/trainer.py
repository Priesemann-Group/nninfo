from typing import Optional
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

import nninfo
from nninfo.config import CLUSTER_MODE
from nninfo.exp_comp import ExperimentComponent
from nninfo.model.quantization import quantizer_list_factory

log = nninfo.logger.get_logger(__name__)

# optimizers for pytorch
OPTIMIZERS_PYTORCH = {"SGD": optim.SGD, "Adam": optim.Adam}

# the losses that are available at the moment
LOSSES_PYTORCH = {
    "BCELoss": nn.BCELoss,
    "CELoss": nn.CrossEntropyLoss,
    "MSELoss": nn.MSELoss,
}

def kl_loss(pred, target):
    rho_hat = torch.mean(torch.abs(pred))
    kl = target * (target / rho_hat).log() + (1 - target) * torch.log((1 - target) / (1 - rho_hat))
    #print(f'rho: {target}, rho_hat: {rho_hat}, kl: {kl}')
    return kl

REGULARIZERS_PYTORCH = {
    "L1": lambda kwargs: (lambda x: torch.mean(torch.abs(x))),
    "L2": lambda kwargs: (lambda x: torch.mean(x**2)),
    "KL": lambda kwargs: (lambda x: kl_loss(x, kwargs['target'])),
}


class Trainer(ExperimentComponent):
    """
    Trains the network using chapter structure.
    Define your training settings here.
    """

    def __init__(
        self,
        dataset_name,
        optim_str,
        loss_str,
        lr,
        shuffle,
        batch_size,
        quantizer,
        momentum=0,
        reg_str=None,
        reg_args=None,
    ):
        """
        Sets training parameters. Is also called when loading parameters from file.
        Args:
            dataset_name (str): Name of the dataset in the TaskManagers dataset
                dict that should be trained on.
            optim_str (str): One of the optimizers available in constant OPTIMIZERS_PYTORCH.
                It is easy to add new ones, if necessary, since most commonly used ones are
                already implemented in pytorch.
            loss_str (str): One of the losses available in LOSSES_PYTORCH.
                It is easy to add new ones, if necessary, since most commonly used ones are
                already implemented in pytorch.
            lr (float): The learning rate that should be used for the training.
            shuffle (bool): Whether to shuffle
            batch_size (int): Number of samples from the dataset that should be used together
                as a batch for one training step, for example in (Batch) Stochastic Gradient
                Descent.
            n_epochs_chapter (int): If the number of epochs per chapter is a constant it can
                be also set here. Otherwise it must be passed each time train_chapter is
                called.
        """

        self._dataset_name = dataset_name
        self._lr = lr
        self._batch_size = batch_size
        self._optim_str = optim_str
        self._loss_str = loss_str
        self._shuffle = shuffle
        self._quantizer_params = quantizer
        self._momentum = momentum

        self._reg_str = reg_str
        self._reg_args = reg_args

        self._n_epochs_trained = 0
        self._n_chapters_trained = 0

        super().__init__()

    @staticmethod
    def from_config(config):
        print(f"config: {config}")
        trainer = Trainer(
            dataset_name=config["dataset_name"],
            optim_str=config["optim_str"],
            loss_str=config["loss_str"],
            lr=config["lr"],
            shuffle=config["shuffle"],
            batch_size=config["batch_size"],
            quantizer=config.get("quantizer", None),
            momentum=config.get("momentum", 0),
            reg_str=config.get("reg_str", None),
            reg_args=config.get("reg_args", None),
        )

        return trainer

    def to_config(self):
        param_dict = {
            "dataset_name": self._dataset_name,
            "optim_str": self._optim_str,
            "batch_size": self._batch_size,
            "shuffle": self._shuffle,
            "lr": self._lr,
            "loss_str": self._loss_str,
            "n_epochs_trained": self._n_epochs_trained,
            "n_chapters_trained": self._n_chapters_trained,
            "quantizer": self._quantizer_params,
            "momentum": self._momentum,
            "reg_str": self._reg_str,
            "reg_args": self._reg_args,
        }

        return param_dict

    @property
    def loss(self):
        return self._loss

    @property
    def lr(self):
        return self._lr

    @property
    def n_epochs_trained(self):
        return self._n_epochs_trained

    @property
    def n_chapters_trained(self):
        return self._n_chapters_trained

    def set_n_epochs_trained(self, n_epochs_trained):
        """
        Sets the number of epochs trained to a new value.
        Should not be called by user, only by experiment.
        """
        log.info("n_epochs_trained is changed from outside.")
        self._n_epochs_trained = n_epochs_trained

    def set_n_chapters_trained(self, n_chapters_trained):
        """
        Sets the number of epochs trained to a new value.
        Should not be called by user, only by experiment.
        """
        log.info("n_chapters_trained is changed from outside.")
        self._n_chapters_trained = n_chapters_trained

    def optimizer_state_dict(self):
        return self._optimizer.state_dict()

    def load_optimizer_state_dict(self, opt_state_dict):
        self._optimizer.load_state_dict(opt_state_dict)

    def regularize(self, activations):
        """
        Regularize the network.

        Args:
            activations (dict): Dictionary containing the activations of the network.
        """

        layerwise = self._reg_args.get("layerwise", False)

        if not layerwise:
            # Combine all activations into one big array
            activations = {"L0": torch.concatenate([activations[key].flatten() for key in activations])}

        regularization_loss = 0
        for values in activations.values():
            regularization_loss += self._regularizer(values)

        return regularization_loss

    def train_chapter(
        self, use_cuda, use_ipex, n_epochs_chapter=None, chapter_ends=None, compute_test_loss: Optional[bool]=True
    ):
        """
        Perform the training steps for a given number of epochs. If no n_epochs_chapter is given
        it is expected to have already been set in set_training_parameters(..).
        Args:
            n_epochs_chapter (int):    Number of epochs to train for this chapter of the training.
            compute_test_loss (bool):  Whether to compute the test loss after each epoch. When None is passed,
                                        it is set to not CLUSTER_MODE.
        """

        if compute_test_loss is None:
            compute_test_loss = not CLUSTER_MODE

        # make experiment components ready for training
        self._start_chapter(use_ipex)
        # set model to train mode
        self._net.train()

        if use_cuda and not next(self._net.parameters()).is_cuda:
            print('Moving model to CUDA')
            print(f'Cuda available: {torch.cuda.is_available()}')
            print(f'Cuda device count: {torch.cuda.device_count()}')
            print(f'Cuda device: {torch.cuda.current_device()}')
            self._net.cuda()

        # create a DataLoader that then feeds the chosen dataset into the network during training
        feeder = DataLoader(
            self._task[self._dataset_name],
            batch_size=self._batch_size,
            shuffle=self._shuffle,
        )

        # central training loop
        for _ in range(n_epochs_chapter):
            full_set_loss = 0
            full_set_reg_loss = 0

            for batch_num, (local_x_batch, local_y_batch) in enumerate(feeder):

                if use_cuda:
                    local_x_batch = local_x_batch.cuda()
                    local_y_batch = local_y_batch.cuda()

                # zeroes the gradient buffers of all parameters
                self._optimizer.zero_grad()

                self._net.train()
                res = self._net.forward(local_x_batch, quantizers=self._quantizer, return_activations_torch=self._reg_str is not None)
                
                if self._reg_str:
                    pred_y, activations = res
                else:
                    pred_y = res

                #print(f"pred_y: {pred_y.shape}, min: {pred_y.min()}, max: {pred_y.max()}")
                #print(f"local_y_batch: {local_y_batch.shape}, min: {local_y_batch.min()}, max: {local_y_batch.max()}")
                loss = self._loss(pred_y, local_y_batch)

                if self._reg_str:
                    reg_loss = self.regularize(activations)
                    total_loss = loss + self._reg_args['weight'] * reg_loss
                else:
                    total_loss = loss

                total_loss.backward()

                self._optimizer.step()

                full_set_loss += loss.cpu().item() * len(local_y_batch)
                full_set_reg_loss += reg_loss.cpu().item() * len(local_y_batch) if self._reg_str else 0

                # Check if chapter ends
                frac_epoch = self.n_epochs_trained + (batch_num + 1) / len(feeder)
                prev_frac_epoch = self.n_epochs_trained + batch_num / len(feeder)

                if batch_num == len(feeder) - 1:
                    self._n_epochs_trained += 1

                if frac_epoch >= chapter_ends[self._n_chapters_trained + 1] and prev_frac_epoch < chapter_ends[self._n_chapters_trained + 1]:
                    self._end_chapter()
                    while self._n_chapters_trained+1 < len(chapter_ends) and frac_epoch >= chapter_ends[self._n_chapters_trained + 1]:
                        self._end_chapter()

            print_str = (
                "trained epoch: "
                + str(self._n_epochs_trained)
                + "; train loss: "
                + str((full_set_loss+((self._reg_args['weight'] * full_set_reg_loss) if self._reg_str is not None else 0)) / len(feeder.dataset))
                + (f" = loss {full_set_loss / len(feeder.dataset)} + {self._reg_args['weight']} * reg {full_set_reg_loss / len(feeder.dataset)}" if self._reg_str else "")
                + (f"; test loss: {self._tester.compute_loss_and_accuracy(quantizer=self._quantizer)[0]}" if compute_test_loss else "")
            )
            print(print_str)
            log.info(print_str)

    def _start_chapter(self, use_ipex=False):

        first_overall_epoch = self._n_epochs_trained == 0 and self.parent.run_id == 0
        first_epoch_in_run = self._n_epochs_trained == 0
        if first_overall_epoch:
            self.initialize_components(use_ipex)
            self.parent.save_components()
        if first_epoch_in_run:
            self.parent.save_checkpoint()

        log.info("Started training chapter {}.".format(self._n_chapters_trained + 1))

    def initialize_components(self, use_ipex=False):

        self._net = self.parent.network

        if torch.cuda.is_available():
            self._net.cuda()
            
        if self._optim_str == "SGD":
            self._optimizer = OPTIMIZERS_PYTORCH[self._optim_str](
                self._net.parameters(), lr=self._lr, momentum=self._momentum
            )
        else:
            self._optimizer = OPTIMIZERS_PYTORCH[self._optim_str](
                self._net.parameters(), lr=self._lr
            )
        self._loss = LOSSES_PYTORCH[self._loss_str]()
        if self._reg_str:
            reg_args = self._reg_args.copy()
            del reg_args["weight"]
            self._regularizer = REGULARIZERS_PYTORCH[self._reg_str](self._reg_args)

        self._task = self.parent.task
        self._tester = self.parent.tester
        self._quantizer = quantizer_list_factory(self._quantizer_params, self.parent.network.get_limits_list())

        if use_ipex:
            import intel_extension_for_pytorch as ipex # type: ignore
            self._net, self._optimizer = ipex.optimize(self._net, optimizer=self._optimizer)

    def _end_chapter(self):
        self._n_chapters_trained += 1
        log.info("Finished training chapter {}.".format(self._n_chapters_trained))
        print("Finished training chapter {}.".format(self._n_chapters_trained))
        self.parent.save_checkpoint()