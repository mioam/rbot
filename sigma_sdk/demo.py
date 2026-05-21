import sigma7
import time
import numpy as np

print(sigma7.drdOpen())
print(sigma7.dhdErrorGetLastStr())
print(sigma7.drdAutoInit())
print(sigma7.dhdErrorGetLastStr())
print(sigma7.drdStart())
print(sigma7.dhdErrorGetLastStr())

sigma7.drdRegulatePos(on=False)
sigma7.drdRegulateRot(on=False)
sigma7.drdRegulateGrip(on=False)

while True:
    print('test: ', sigma7.drdGetPositionAndOrientation())
    time.sleep(0.1)
