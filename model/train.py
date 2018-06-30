import os
import sys

sys.path.append(os.path.abspath("../"))

from model.xception import model21_val2
from model.xception import model22_val2
from model.xception import model23_val2

model21_val2.train()
model22_val2.train()
model23_val2.train()
