from abc import ABC

import nninfo

log = nninfo.logger.get_logger(__name__)


class ExperimentComponent(ABC):
    """
    Abstract class that defines parent property for each component of the experiment
    (experiment is then the parent of each component, if they are connected).
    """

    def __init__(self):
        self._parent = None
        super().__init__()

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        if self.parent is not None:
            if parent is not None:
                log.warning(
                    "Parent of {} is changed to experiment {}.".format(
                        type(self)),
                    parent.id,
                )
            else:
                log.info("Parent of {} is removed.".format(type(self)))
        self._parent = parent
