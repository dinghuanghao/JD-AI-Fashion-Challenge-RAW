import os
import sys
from time import sleep
sys.path.append(os.path.abspath("../"))

from model.densenet201 import model32_val3, model32_val4
model32_val3.train()
sleep(300)
model32_val4.train()