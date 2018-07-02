import os
import sys
from time import sleep
sys.path.append(os.path.abspath("../"))

from model.densenet201 import model29_val3, model29_val4, model30_val3, model30_val4, model31_val3, model31_val4, \
    model32_val3, model32_val4

model29_val3.train()
model29_val4.train()
sleep(120)
model30_val3.train()
model30_val4.train()
sleep(120)
model31_val3.train()
model31_val4.train()
sleep(120)
model32_val3.train()
model32_val4.train()