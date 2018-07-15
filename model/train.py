import os
import sys
from time import sleep
sys.path.append(os.path.abspath("../"))

from model.inceptionresnetv2 import model16_val4
model16_val4.train()