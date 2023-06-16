
from PIL import Image
import numpy as np
import random
import copy
import os
import io




imgpath="E:\\海漂垃圾自动识别\\样本集\\全部垃圾提取\\秀屿区东埔镇塔林岸段img"
imgpath1="E:\\海漂垃圾自动识别\\样本集\\全部垃圾提取\\秀屿区东埔镇塔林岸段label"
imgs = os.listdir(imgpath1)

for jpg in imgs:
    path=imgpath+"\\"+jpg
    path1=imgpath1+"\\"+jpg
    ispositive=os.path.exists(path)
    print(ispositive)
   
    if ispositive:
        pass
    else:
        try:
            os.remove(path1)
        except:
            pass
