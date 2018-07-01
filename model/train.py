import os
import sys

sys.path.append(os.path.abspath("../"))

from model.xception import model23_val2
from model.nasmobile import model5_val1

model23_val2.train()
model5_val1.train()
