
from PIL import Image
import numpy as np
import random
import copy
import os
import io



def visiblelabel(imgpath,labelpath,psavepath):
    
    imgs = os.listdir(labelpath)
    num=len(imgs)
    i=0
   
    for jpg in imgs:
        try:
            img=Image.open(imgpath+"\\"+jpg)
            label = Image.open(labelpath+"\\"+jpg)
            orininal_h = np.array(label).shape[0]
            orininal_w = np.array(label).shape[1]

            label = np.array(label)*255
            seg_img=np.zeros((orininal_h,orininal_w,3))
            seg_img[:,:,0]=label

            label=Image.fromarray(np.uint8(seg_img)).resize((orininal_h,orininal_w))
            
    
            
        
        

            image = Image.blend(img,label,0.3)
        

            image.save(psavepath+"\\"+jpg,'TIFF')
            i=i+1
            print("已执行"+str(i/num*100)+"%")
        except:
            pass
imgpath=r"Y:\wsw\wsw2\海漂垃圾\泡沫浮球样本库\蕉城区漳湾镇拱屿岸段img"
labelpath=r"Y:\wsw\wsw2\海漂垃圾\泡沫浮球样本库\蕉城区漳湾镇拱屿岸段label"
psavepath=r"Y:\wsw\wsw2\海漂垃圾\泡沫浮球样本库\visiblelabel"

visiblelabel(imgpath,labelpath,psavepath)