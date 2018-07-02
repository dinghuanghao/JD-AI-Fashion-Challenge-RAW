from model.resnet50 import model16_val3, model16_val4, model17_val3, model17_val4
from time import sleep

model16_val3.train()
sleep(300)
model16_val4.train()
sleep(300)
model17_val3.train()
sleep(300)
model17_val4.train()