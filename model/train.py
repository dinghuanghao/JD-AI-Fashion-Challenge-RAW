import os
import sys
from time import sleep
sys.path.append(os.path.abspath("../"))

# from model.inceptionresnetv2 import model16_val4
# model16_val4.train()

# from model.inceptionv3 import model21_val3, model23_val3
# model21_val3.train()
# model23_val3.train()

from model.resnet50 import model16_val3
model16_val3.train()