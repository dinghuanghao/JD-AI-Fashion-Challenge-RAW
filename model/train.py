from model.densenet169 import model4_val5 as dense169_65
from model.inceptionresnetv2 import model7_val5 as inres_65
from model.vgg19 import model6_val5 as vgg_65

dense169_65.train()
inres_65.train()
vgg_65.train()