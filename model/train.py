import os
import sys
from time import sleep
sys.path.append(os.path.abspath("../"))

from model.densenet169 import model111_val2
model111_val2.train()

from model.resnet50 import model111_val1
model111_val1.train()

from model.xception import model111_val1
model111_val1.train()
