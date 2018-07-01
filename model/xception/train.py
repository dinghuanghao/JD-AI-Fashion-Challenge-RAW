from model.xception import model21_val3
from model.xception import model21_val4
from model.xception import model22_val3
from time import sleep

model21_val3.train()
sleep(300)
model21_val4.train()
sleep(300)
model22_val3.train()
