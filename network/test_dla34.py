import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import os
import sys
import tensorboardX
import dla34


heads = {'hm': 7, 'box_size': 3, 'depthmap': 1, 'orientation': 3}
the_dla34 = dla34.DLASeg('dla34', heads, '../model/')