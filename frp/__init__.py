import pathlib
import sys

CURRENT_DIR = pathlib.Path(__file__).absolute().parent
sys.path = [str(CURRENT_DIR)] + sys.path  # avoid using the same 'utils.py'
