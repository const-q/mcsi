import sys
import torch
from .mcsi import mcsi

sys.modules['mcsi'] = mcsi