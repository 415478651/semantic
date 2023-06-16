import os
import gdal
import numpy as np
import time
import rasterio
from rasterio import crs
from rasterio.windows import Window
from multiprocessing import Pool

#  读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    return dataset
def Union(a,b,new_name2,bias,box):
        imgae=rasterio.open(new_name2)
        h=imgae.read(1,window=Window(bias,bias,box,box))
        g=imgae.read(2,window=Window(bias,bias,box,box))
        r=imgae.read(3,window=Window(bias,bias,box,box))
        img.write(h,indexes=1,window=Window(b, a, box, box))
        img.write(g,indexes=2,window=Window(b, a, box, box))
        img.write(r,indexes=3,window=Window(b, a, box, box))

#  保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

def Crop(a,b,c,d,new_name):
    #with rasterio.open(new_name) as imgage:
        #b, g, r = (imgage.read(k) for k in (1, 2, 3))
    b=np.ones((512, 512), dtype=rasterio.ubyte) * 127
    

    
    with rasterio.open(TifPath,'w',driver='GTiff',width=38476,height=33929,count=1,dtype=b.dtype,compress='lzw') as img:
        #for k, arr in [(1, b), (2, g), (3, r)]:
        img.write(b.astype(rasterio.uint8),indexes=1,window=Window(a, b, 512, 512))
        img.close()
        #img.write(g,indexes=2,window=Window(a, b, c, d))
        #img.write(r,indexes=3,window=Window(a, b, c, d))
        #bands=r.shape
        #if (len(bands) == 2):
        #
        #    cropped = img[
        #              a: b,
        #              c: d]
        #    #  如果图像是多波段
        #else:
        #    cropped = img[:,
        #              a: b,
        #              c: d]
    #writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif" % new_name)
    #writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif" % new_name)


'''
滑动窗口裁剪函数
TifPath 影像路径
SavePath 裁剪后保存目录
CropSize 裁剪尺寸
RepetitionRate 重复率
'''
TifPath1=r"F:\tensorflow_classification\efficientnetV2\test2019\c2020年4月南安市影像_Clip1.tif"
TifPath=r"F:\tensorflow_classification\efficientnetV2\testout2020\test.tif"
SavePath=r"F:\tensorflow_classification\efficientnetV2\testout2020label"
CropSize=512
RepetitionRate=0
#dataset_img = readTif(TifPath)
dataset_img = rasterio.open(TifPath1)
width = dataset_img.width
height = dataset_img.height
crs = dataset_img.crs

#proj = dataset_img.GetProjection()
#geotrans = dataset_img.GetGeoTransform()
#img = dataset_img.ReadAsArray(0, 0, width, height)  # 获取数据
#  获取当前文件夹的文件个数len,并以len+1命名即将裁剪得到的图像
new_name = 1

new_name2=SavePath + "/%d.tif" % new_name
img=rasterio.open(TifPath,'w',driver='GTiff',width=width,height=height,count=3,dtype=rasterio.uint8,crs=crs,transform=dataset_img.transform) 


if __name__ == '__main__':
        t1 = time.time()



        param=[]

        #for k, arr in [(1, b), (2, g), (3, r)]:
        heightrange=int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))
        widthrange=int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))
        box=CropSize * (1 - RepetitionRate)
        bias=CropSize * RepetitionRate/2
        
        for i in range(heightrange):
            for j in range(widthrange):
                #  如果图像是单波段
                a=int(i * CropSize * (1 - RepetitionRate))+bias

                b=int(j * CropSize * (1 - RepetitionRate))+bias

               
	
                t=(a,b,new_name2,bias,box)
                param.append(t)
                new_name = new_name + 1
                new_name2=SavePath + "/%d.tif" % new_name


        print(param)
        p=Pool(processes=1)
        b=p.starmap(Union,param)
        p.close()
        p.join()	
        t2 = time.time()
        print(t2-t1)
  


