import os
import gdal
import numpy as np
import time
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
    if (len(img.shape) == 2):
        cropped = img[
                  a: b,
                  c: d]
        #  如果图像是多波段
    else:
        cropped = img[:,
                  a: b,
                  c: d]
    #writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif" % new_name)
    #new_name="img"+str(new_name)
    writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif" % new_name)


'''
滑动窗口裁剪函数
TifPath 影像路径
SavePath 裁剪后保存目录
CropSize 裁剪尺寸
RepetitionRate 重复率
'''
TifPath=r"F:/南安三调/南安市六期影像成果及镶嵌块/2021年2月南安市6期影像.img"
SavePath=r"testout2020label"
CropSize=512
RepetitionRate=0
dataset_img = readTif(TifPath)
width = dataset_img.RasterXSize
height = dataset_img.RasterYSize
proj = dataset_img.GetProjection()
geotrans = dataset_img.GetGeoTransform()
img = dataset_img.ReadAsArray(0, 0, width, height)  # 获取数据
#  获取当前文件夹的文件个数len,并以len+1命名即将裁剪得到的图像
#new_name = len(os.listdir(SavePath)) + 1
new_name=40000
if __name__ == '__main__':
    t1 = time.time()


    #  裁剪图片,重复率为RepetitionRate
    param=[]

    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
                #  如果图像是单波段
            a=int(i * CropSize * (1 - RepetitionRate))
            b=a + CropSize
            c=int(j * CropSize * (1 - RepetitionRate))
            d=c + CropSize
            t=(a,b,c,d,new_name)


                #  写图像
            param.append(t)
                #  文件名 + 1
            new_name = new_name + 1
    p = Pool(processes=6)
    b = p.starmap(Crop, param)
    p.close()
    p.join()

    #  向前裁剪最后一列
    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        if (len(img.shape) == 2):
            cropped = img[int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                      (width - CropSize): width]
        else:
            cropped = img[:,
                      int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                      (width - CropSize): width]
            #  写图像
        writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif" % new_name)
        new_name = new_name + 1
        #  向前裁剪最后一行
    for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        if (len(img.shape) == 2):
            cropped = img[(height - CropSize): height,
                        int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        else:
            cropped = img[:,
                        (height - CropSize): height,
                          int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif" % new_name)
            #  文件名 + 1
        new_name = new_name + 1
        #  裁剪右下角
    if (len(img.shape) == 2):
        cropped = img[(height - CropSize): height,
                  (width - CropSize): width]
    else:
        cropped = img[:,
                (height - CropSize): height,
                (width - CropSize): width]
    writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif" %new_name)
    new_name = new_name + 1


    #  将影像1裁剪为重复率为0.1的256×256的数据集


    t2 = time.time()
    print(t2-t1)
