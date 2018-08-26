import os
import sys
from time import sleep
sys.path.append(os.path.abspath("../"))

# from model.xception import model120_val1
# model120_val1.train()
# from model.densenet169 import model120_val1
# model120_val1.train()
from model.resnet50 import model119_val1
model119_val1.train()