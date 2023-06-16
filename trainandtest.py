from PIL import Image
import numpy as np
import random
import copy
import os
import io



def positive(imgpath,labelpath,Timgsavepath,Tlabelsavepath,Vlabelsavepath,Vimgsavepath):
    
    imgs = os.listdir(labelpath)
    num=len(imgs)
    numt=int(num*0.1)
    i=0
    np.random.seed(10011)
    np.random.shuffle(imgs)

 
    for jpg in imgs[0:numt]:
        try:
            label = Image.open(labelpath+"\\"+jpg)
            img = Image.open(imgpath+"\\"+jpg)
            i=str(len(os.listdir(Timgsavepath)))
            print(i)
            label.save(Tlabelsavepath+"\\"+i+".tif","TIFF")
            img.save(Timgsavepath+"\\"+i+".tif","TIFF")
            label=None
            img=None
           
            
        except:
            print("false")
    for jpg in imgs[numt:]:
        try:
            label = Image.open(labelpath+"\\"+jpg)
            img = Image.open(imgpath+"\\"+jpg)
            i=str(len(os.listdir(Vlabelsavepath)))
            print(i)
            label.save(Vlabelsavepath+"\\"+i+".tif","TIFF")
            
            img.save(Vimgsavepath+"\\"+i+".tif","TIFF")
            label=None
            img=None
           
            
        except:
            print("false")
            

imgpath=r"Y:\wsw\wsw2\海漂垃圾\泡沫浮球样本库\平潭苏澳镇苏澳至斗魁岸段img"
labelpath=r"Y:\wsw\wsw2\海漂垃圾\泡沫浮球样本库\平潭苏澳镇苏澳至斗魁岸段label"
Timgsavepath=r"Y:\wsw\wsw2\海漂垃圾\泡沫浮球样本库\testimg"
Tlabelsavepath=r"Y:\wsw\wsw2\海漂垃圾\泡沫浮球样本库\testlabel"
Vimgsavepath=r"Y:\wsw\wsw2\海漂垃圾\泡沫浮球样本库\trainimg"
Vlabelsavepath=r"Y:\wsw\wsw2\海漂垃圾\泡沫浮球样本库\trainlabel"
positive(imgpath,labelpath,Timgsavepath,Tlabelsavepath,Vlabelsavepath,Vimgsavepath)