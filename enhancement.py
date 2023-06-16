from PIL import Image,ImageEnhance
import os
import io
import numpy as np

#色调增强
def colorenhance(img,r=0.8,save=False,savepath):
    img_color=ImageEnhance.Color(img)
    img_color=img_color.enhance(r)
    if save:
        img_color.save(savepath)
    return img_color
#对比度增强
def contrastenhance(img,r=0.8,save=False,savepath):
    img_contrast=ImageEnhance.Contrast(img)
    img_contrast=img_contrast.enhance(r)
    if save:
        img_color.save(savepath)
    return img_contrast
#亮度增强
def brightenhance(img,r=1.5,save=False,savepath):
    img_bright=ImageEnhance.Brightness(img)
    img_bright=img_bright.enhance(r)
    if save:
        img_color.save(savepath)
    return img_bright
#色调自动调节
def autocolorenhance(img,save=False,savepath,trb=1.4790572475583728,trg=1.0665602806363264):
    
    img_merge = np.array(img)
    r=np.sum(img_merge[:,:,0])
    g=np.sum(img_merge[:,:,1])
    b=np.sum(img_merge[:,:,2])
    r_b=trb/(b/r)
    r_g=trg/(b/g)
    img_merge[:,:,0]=img_merge[:,:,0]/r_b
    img_merge[:,:,1]=img_merge[:,:,1]/r_g
    if save:
        img_merge = Image.fromarray(np.uint8(img_merge)).resize((512,512))
        img_merge.save(savepath)
    return img_merge
#水平翻转
def fliph(img,save=False,savepath):
    img=img.transpose(Image.FLIP_LEFT_RIGHT)
    if save:
        img.save(savepath)
    return img
#垂直翻转
def flipv(img,save=False,savepath):
    img=img.transpose(Image.FLIP_TOP_BOTTOM)
    if save:
        img.save(savepath)
    return img
