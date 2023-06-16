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
TifPath1=r"shuitou.tif"
TifPath=r"test.tif"
SavePath=r"shuitou"
CropSize=512
RepetitionRate=0.5
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
if __name__ == '__main__':
    t1 = time.time()


    #  裁剪图片,
    param=[]
    with rasterio.open(TifPath,'w',driver='GTiff',width=width,height=height,count=3,dtype=rasterio.uint8,crs=crs,transform=dataset_img.transform) as img:
        #for k, arr in [(1, b), (2, g), (3, r)]:
        heightrange=int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))
        widthrange=int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))
        box=CropSize * (1 - RepetitionRate)
        bias=CropSize * RepetitionRate/2
        for i in range(heightrange):
            for j in range(widthrange):
                #  如果图像是单波段
                a=int(i * CropSize * (1 - RepetitionRate))+bias
                #b=a + CropSize
                b=int(j * CropSize * (1 - RepetitionRate))+bias
                #with rasterio.open(new_name2) as imgage:
                    #b, g, r = (imgage.read(k) for k in (1, 2, 3))
                imgae=rasterio.open(new_name2)
                h=imgae.read(1,window=Window(bias,bias,box,box))
                g=imgae.read(2,window=Window(bias,bias,box,box))
                r=imgae.read(3,window=Window(bias,bias,box,box))
                
                #bb=np.ones((512, 512), dtype=rasterio.uint8) * 127
                #d=c + CropSize
                
                
                img.write(h,indexes=1,window=Window(b, a, box, box))
                img.write(g,indexes=2,window=Window(b, a, box, box))
                img.write(r,indexes=3,window=Window(b, a, box, box))
                new_name = new_name + 1
                new_name2=SavePath + "/%d.tif" % new_name
                #img.close()
                #Crop(b,a,CropSize,CropSize,new_name2)


                #写图像
                #param.append(t)
                #文件名 + 1
                
    #print(param)
    #p = Pool(1)
    #b = p.starmap(Crop, param)
    #p.close()
    #p.join()
    t2 = time.time()
    print(t2-t1)
    ''''#  向前裁剪最后一列
    #for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
    #    if (len(img.shape) == 2):
    #        cropped = img[int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
    #                  (width - CropSize): width]
    #    else:
    #        cropped = img[:,
    #                  int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
    #                  (width - CropSize): width]
    #        #  写图像
    #    writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif" % new_name)
    #    new_name = new_name + 1
    #    #  向前裁剪最后一行
    #for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
    #    if (len(img.shape) == 2):
    #        cropped = img[(height - CropSize): height,
    #                    int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
    #    else:
    #        cropped = img[:,
    #                    (height - CropSize): height,
    #                      int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
    #    writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif" % new_name)
    #        #  文件名 + 1
    #    new_name = new_name + 1
    #    #  裁剪右下角
    #if (len(img.shape) == 2):
    #    cropped = img[(height - CropSize): height,
    #              (width - CropSize): width]
    #else:
    #    cropped = img[:,
    #            (height - CropSize): height,
    #            (width - CropSize): width]
    #writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif" % new_name)
    #new_name = new_name + 1


    #  将影像1裁剪为重复率为0.1的256×256的数据集'''



