from model.densenet121 import model15_val1
from model.densenet201 import model4_val1
from model.inceptionresnetv2 import model5_val1
from model.inceptionv3 import model18_val1
from model.vgg19 import model4_val1 as vgg4

model15_val1.train()
model4_val1.train()
model5_val1.train()
model18_val1.train()
vgg4.train()