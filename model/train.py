import os
import sys
from time import sleep
sys.path.append(os.path.abspath("../"))

from model.xception import model27_val3, model27_val4, model28_val3, model29_val2
from model.xception import model29_val3, model29_val4, model30_val2, model30_val5

model27_val4.train()
model28_val3.train()
model29_val2.train()
model29_val3.train()
model29_val4.train()
model30_val5.train()
model30_val2.train()
