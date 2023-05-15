import logging
import os

from . import logger
from . import config
from . import experiment
from . import trainer
from . import tester
from . import tasks
from . import file_io
from . import plot
from . import data_set
from . import analysis
from . import schedule

from .experiment import *
from .trainer import *
from .tester import *
from .model.neural_network import *
from .tasks import *

module_dir = os.path.dirname(__file__) + "/"  # path to this file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-13.13s] [%(levelname)-5.5s]  %(message)s",
    filename=module_dir + "../experiments/nninfo.log",
    filemode="a",
)
logging.getLogger("nninfo").info("STARTUP NNINFO SESSION")
