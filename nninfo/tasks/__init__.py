import os
import glob

# Get a list of all Python module files in the current directory
modules = glob.glob(os.path.dirname(__file__) + "/*.py")
__all__ = [os.path.basename(f)[:-3] for f in modules if os.path.isfile(f) and not f.startswith('_')]

#for f in os.listdir(os.path.dirname(__file__)):
#    if os.path.isfile(os.path.join(os.path.dirname(__file__), f)) and not f.startswith('_'):
#        exec("from . import {}".format(f[:-3]))
