from PIL import Image
import numpy as np
import random
import copy
import os
import io



def positive(imgpath,labelpath,psavepath,nsavepath):
    
    imgs = os.listdir(imgpath)
    num=len(imgs)
    i=0
   
    for jpg in imgs:

        label = Image.open(labelpath+"\\"+jpg)
        orininal_h = np.array(label).shape[0]
        orininal_w = np.array(label).shape[1]

        label = np.array(label)
        ispositive=np.any(label)
        label=Image.fromarray(np.uint8(label)).resize((orininal_w,orininal_h))
        img=Image.open(imgpath+"\\"+jpg)
        i=i+1
    
        if ispositive:
            psavepaths = os.listdir(psavepath+"\\img")
            n=len(psavepaths)+1
            img.save(psavepath+"\\img\\"+str(n)+".tif",'TIFF')
            label.save(psavepath+"\\label\\"+str(n)+".tif",'TIFF')
        else:
            nsavepaths = os.listdir(nsavepath+"\\img")
            n=len(nsavepaths)+1
            img.save(nsavepath+"\\img\\"+str(n)+".tif",'TIFF')
            label.save(nsavepath+"\\label\\"+str(n)+".tif",'TIFF')
        print("已执行"+str(i/num*100)+"%")
imgpath="E:\\海漂垃圾自动识别\\样本集\\废弃撑架\\同安区西柯镇丙洲社区岸段img2"
labelpath="E:\\海漂垃圾自动识别\\样本集\\废弃撑架\\同安区西柯镇丙洲社区岸段label2"
psavepath="E:\\海漂垃圾自动识别\\样本集\\废弃撑架\\positive"
nsavepath="E:\\海漂垃圾自动识别\\样本集\\废弃撑架\\negtive"
positive(imgpath,labelpath,psavepath,nsavepath)