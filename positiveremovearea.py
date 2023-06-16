from PIL import Image
import numpy as np
import random
import copy
import os
import io



def positive(imgpath,labelpath,area):
    
    imgs = os.listdir(labelpath)
    num=len(imgs)
    i=0
   
    for jpg in imgs:
        try:
            label = Image.open(labelpath+"\\"+jpg)
            #orininal_h = np.array(label).shape[0]
            #orininal_w = np.array(label).shape[1]

            label = np.array(label)
            ispositive=np.sum(label)
            #label=Image.fromarray(np.uint8(label)).resize((orininal_w,orininal_h))
            #img=Image.open(imgpath+"\\"+jpg)
            i=i+1
            label=None
            print(ispositive)
            if ispositive>area:
                pass
            else:

                #os.remove(imgpath+"\\"+jpg)
                os.remove(imgpath+"\\"+jpg)
                print("remove")
            #print("已执行"+str(i/num*100)+"%")
        except:
            print("false")

imgpath=r"E:\海漂垃圾自动识别\样本集\废弃撑架\positive\visiblelabel"
labelpath=r"E:\海漂垃圾自动识别\样本集\废弃撑架\positive\label256"
area=200
positive(imgpath,labelpath,area)