
from PIL import Image
import numpy as np
import random
import copy
import os
import io




imgpath=r"Y:\wsw\wsw2\海漂垃圾\泡沫浮球样本库\连江县马鼻镇合丰岸段img"
imgpath1=r"Y:\wsw\wsw2\海漂垃圾\泡沫浮球样本库\连江县马鼻镇合丰岸段label"
imgs = os.listdir(imgpath1)

imgs2= os.listdir(imgpath)
print(imgs)
a=[]
i=0

for jpg in imgs2:
    if jpg in imgs:
       pass
    else:
       os.remove(imgpath+"\\"+jpg)
    