from model.inceptionv3 import model21_val3
from model.inceptionv3 import model21_val4
from model.inceptionv3 import model22_val3
from model.inceptionv3 import model23_val3
from time import sleep

model21_val3.train()
sleep(300)
model21_val4.train()
sleep(300)
model22_val3.train()
sleep(300)
model23_val3.train()