from PIL import Image
import numpy as np
import random
import copy
import os
import io



def positive(imgpath,labelpath):
    
    imgs = os.listdir(imgpath)
    num=len(imgs)
    i=0
   
    for jpg in imgs:
        try:
            label = Image.open(labelpath+"\\"+jpg)
            orininal_h = np.array(label).shape[0]
            orininal_w = np.array(label).shape[1]

            label = np.array(label)
            ispositive=np.any(label)
            label=Image.fromarray(np.uint8(label)).resize((orininal_w,orininal_h))
            #img=Image.open(imgpath+"\\"+jpg)
            i=i+1
            #img=None
            label=None

            if ispositive:
                pass
            else:

                os.remove(imgpath+"\\"+jpg)
                os.remove(labelpath+"\\"+jpg)
            print("已执行"+str(i/num*100)+"%")
        except:
            print("false")
imgpath=r"Y:\wsw\wsw2\海漂垃圾\泡沫浮球样本库\罗源县碧里乡廪尾岸段img"
labelpath=r"Y:\wsw\wsw2\海漂垃圾\泡沫浮球样本库\罗源县碧里乡廪尾岸段label"

positive(imgpath,labelpath)