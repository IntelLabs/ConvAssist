import importlib
import importlib.metadata

from . import ConvAssist as ConvAssist
from . import predictor as predictor
from . import predictor_activator as predictor_activator
from . import predictor_registry as predictor_registry
from . import utilities as utilities

__version__ = importlib.metadata.version(__package__ or __name__)
