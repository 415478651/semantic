
from PIL import Image
import numpy as np
import random
import copy
import os
import io




imgpath="Y:\\wsw\\wsw2\\海漂垃圾\\泡沫浮球样本库\\平潭苏澳镇苏澳至斗魁岸段img"
imgpath1="Y:\\wsw\\wsw2\\海漂垃圾\\泡沫浮球样本库\\平潭苏澳镇苏澳至斗魁岸段label"
imgs = os.listdir(imgpath)

for jpg in imgs:
    path=imgpath+"\\"+jpg
    path1=imgpath1+"\\"+jpg
    try:
        img = Image.open(imgpath+"\\"+jpg)
        #orininal_h = np.array(img).shape[0]
        #orininal_w = np.array(img).shape[1]

        img = np.array(img)
        ispositive=np.any(img)
        
    
        if ispositive:
            #img.save("F:\\海漂垃圾自动识别\\样本集\\训练集\\positive\\"+jpg,'TIFF')
            pass
        else:
            try:
                print(path)
                os.remove(path)
                os.remove(path1)
            except:
                pass
    except:
                pass