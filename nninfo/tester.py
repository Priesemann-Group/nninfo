import torch
from torch.utils.data import DataLoader
import pandas as pd

from nninfo.exp_comp import ExperimentComponent
from nninfo.model.quantization import quantizer_list_factory


class Tester(ExperimentComponent):
    """
    Is called after each training chapter to perform predefined tests and save their results.
    Args:
        dataset_name (str): Name of the dataset in the TaskManagers dataset
            dict that should be tested on.
    """

    BATCH_SIZE = 10_000

    def __init__(self, dataset_name):
        super().__init__()
        self._dataset_name = dataset_name
        self._net = None
        self._task = None

    @staticmethod
    def from_config(config):
        return Tester(config["dataset_name"])

    def to_config(self):
        return {"dataset_name": self._dataset_name}

    def _get_output_activations(self, dataset_name=None, quantizer_params=None, quantizer=None):
        self._net = self.parent.network
        self._task = self.parent.task

        if quantizer is None:
            quantizer = quantizer_list_factory(
                quantizer_params, self._net.get_limits_list())

        self._net.eval()
        if dataset_name is None:
            dataset_name = self._dataset_name
        feeder = DataLoader(
            self._task[dataset_name], batch_size=self.BATCH_SIZE
        )
        with torch.no_grad():
            for x_test, y_test in feeder:
                yield self._net.forward(x_test, quantizer, apply_output_softmax=True), y_test

    def compute_loss_and_accuracy(self, dataset_name=None, quantizer_params=None, quantizer=None):

        activations_iter = self._get_output_activations(
            dataset_name, quantizer_params, quantizer)

        loss_fn = self.parent.trainer.loss

        total_size = 0
        correct = 0
        test_loss = 0
        for pred_y_test, y_test in activations_iter:

            # Compute loss
            loss = loss_fn(pred_y_test, y_test)
            test_loss += loss.item() * pred_y_test.shape[0]

            # Compute accuracy
            if self.parent.task.task.y_dim > 1 and y_test.ndim == 1:
                # One-hot-representation
                decision = pred_y_test.argmax(dim=1)
                correct += (decision == y_test).sum().item()
            else:
                # Binary output representations
                decision = torch.round(pred_y_test)
                correct += torch.all(decision == y_test, axis=1).sum().item()

            total_size += pred_y_test.shape[0]

        loss = test_loss / total_size
        accuracy = correct / total_size

        return loss, accuracy
    
    def compute_loss_and_accuracy_by_target(self, dataset_name=None, quantizer_params=None, quantizer=None):

        activations_iter = self._get_output_activations(
            dataset_name, quantizer_params, quantizer)
        
        loss_fn = self.parent.trainer.loss

        losses_per_target = {}
        correct_per_target = {}
        count_per_target = {}

        for pred_y_test_batch, y_test_batch in activations_iter:

            for pred_y_test, y_test in zip(pred_y_test_batch, y_test_batch):

                # Compute loss
                loss = loss_fn(pred_y_test, y_test).item()

                # Compute accuracy
                if self.parent.task.task.y_dim > 1 and isinstance(y_test, int):
                    # One-hot-representation
                    decision = pred_y_test.argmax(dim=1)
                    correct = (decision == y_test).item()
                    target = y_test
                else:
                    # Binary output representations
                    decision = torch.round(pred_y_test)
                    correct = torch.all(decision == y_test).item()
                    target = tuple(int(i) for i in y_test)

                # Add loss, accuracy and count
                if target not in losses_per_target:
                    losses_per_target[target] = 0
                    correct_per_target[target] = 0
                    count_per_target[target] = 0

                losses_per_target[target] += loss
                correct_per_target[target] += correct
                count_per_target[target] += 1

        # Compute average loss and accuracy for each class
        class_losses_avg = {target: loss / count for target, (loss, count) in dict_zip(losses_per_target, count_per_target)}
        class_accuracy_avg = {target: correct / count for target, (correct, count) in dict_zip(correct_per_target, count_per_target)}

        return class_losses_avg, class_accuracy_avg
        
def dict_zip(*dicts):
    for key in dicts[0]:
        yield key, tuple(d[key] for d in dicts)