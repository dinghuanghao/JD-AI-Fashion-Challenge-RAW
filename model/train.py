import os
import sys
from time import sleep
sys.path.append(os.path.abspath("../"))

from model.resnet50 import model16_val4 as resnet50_16_4
from model.vgg16 import model4_val4 as vgg16_4_4
from model.vgg19 import model17_val4 as vgg19_7_4
from model.xception import model12_val4 as xception_12_4
from model.xception import model30_val4 as xception_30_4
from model.xception import model31_val4 as xception_31_4

resnet50_16_4.train()
vgg16_4_4.train()
vgg19_7_4.train()
xception_12_4.train()
xception_30_4.train()
xception_31_4.train()