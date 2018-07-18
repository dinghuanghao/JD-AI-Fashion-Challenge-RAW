import os
import sys
from time import sleep
sys.path.append(os.path.abspath("../"))

# from model.xception import model29_val2
# from model.xception import model29_val3, model29_val4, model30_val2, model30_val5
#
# model29_val2.train()
# model29_val3.train()
# model29_val4.train()
# model30_val5.train()
# model30_val2.train()

from model.densenet201 import model37_val3

model37_val3.train()