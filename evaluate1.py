from PIL import Image
import time

import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


ALPHA = 1.0
#$WEIGHTS_PATH_MOBILE = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5"

                    
NCLASSES = 2
HEIGHT = 512    
WIDTH = 512
print("heirenwenhao")
def letterbox_image(image, size, type):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    
    image = image.resize((nw,nh), Image.BICUBIC)
    if(type=="jpg"):
        new_image = Image.new('RGB', size, (0,0,0))
    elif(type=="png"):
        new_image = Image.new('RGB', size, (0,0,0))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image,nw,nh

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_random_data(image, label, input_shape, jitter=.1, hue=.2, sat=1.2, val=1.2):

    h, w = input_shape

    resize = rand()<.5
    if resize: 
    # resize image
        rand_jit1 = rand(1-jitter,1+jitter)
        rand_jit2 = rand(1-jitter,1+jitter)
        new_ar = w/h * rand_jit1/rand_jit2
        scale = rand(.7, 1.3)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.BICUBIC)
        # place image
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (0,0,0))
        new_label = Image.new('L', (w,h), 0)
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label
    # flip image or not
    flip = rand()<.5
    if flip: 
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    #flip = rand()<.5
    if flip:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        label = label.transpose(Image.FLIP_TOP_BOTTOM)

    # distort image
    distort = rand()<.5
    if distort: 
    #$if not flip:
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = rgb_to_hsv(np.array(image)/255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x>1] = 1
        x[x<0] = 0
        image = hsv_to_rgb(x)
        image=image*255
        '''image=Image.fromarray(np.uint8(image)).resize((512,512))
        path='.\\testout4\\'+str(rand(0,5000))+'.tif'
        image.save(path,'TIFF')'''

    return image,label


def generate_arrays_from_file(lines):
    # ��ȡ�ܳ���
    n = lines
    
    i = 0

 
    # ��ȡһ��batch_size��С������
    f1=0
   
    for j in range(1,n):
       
        name = str(j)+'.tif'
        
        # ���ļ��ж�ȡͼ��
        jpg = Image.open(r".\2020水头测试\2020truelabel0.390625" + '/' + name)
        png = Image.open(r".\2020水头测试\2020prelabel0.390625" + '/' + name)
        #png,_,_ = letterbox_image(png,(HEIGHT,WIDTH),"png")

        jpg = np.array(jpg)
        png = np.array(png)
        F11=F1(jpg,png)
        f1=F11+f1
        print(i)
        print(F11)
        i=i+1
    f=f1/n 
    s2=0 
    i=0
    for j in range(1,n):
       
        name = str(j)+'.tif'
        
        # ���ļ��ж�ȡͼ��
        jpg = Image.open(r".\2020水头测试\2020truelabel0.390625" + '/' + name)
        png = Image.open(r".\2020水头测试\2020prelabel0.390625" + '/' + name)
        #png,_,_ = letterbox_image(png,(HEIGHT,WIDTH),"png")

        jpg = np.array(jpg)
        png = np.array(png)
        F11=F1(jpg,png)
        s2=s2+((F11-f)*(F11-f))
       
        i=i+1
    print(f) 
    print(s2/n)

  

def Recall(y_true,y_pred):
    y_true=y_true.flatten()
    
    y_pred=y_pred.flatten()
   
    true_pos=np.sum(y_pred*y_true)
    
    possible_pos=np.sum(y_true)
    if possible_pos==0:
        possible_pos=possible_pos+(1e-10)
    print("possible_pos",possible_pos)
    return true_pos/possible_pos
def Precision(y_true,y_pred):
    y_true=y_true.flatten()
    y_pred=y_pred.flatten()
    true_pos=np.sum(y_pred*y_true)
    pre_pos=np.sum(y_pred)
    if pre_pos==0:
        pre_pos=pre_pos+(1e-10)
    print("pre_pos",pre_pos)
    return true_pos/(pre_pos)
def F1(y_true,y_pred):
    pre=Precision(y_true,y_pred)
    recall=Recall(y_true,y_pred)
    s=pre+recall
    if s==0:
        s=s+(1e-10)
    return 2*(pre*recall)/(s)
def Diceloss(y_true,y_pred):
    f1=F1(y_true,y_pred)
    return (1-f1)
def loss(y_true, y_pred):
    crossloss = K.categorical_crossentropy(y_true,y_pred)
    loss = K.sum(crossloss)/HEIGHT/WIDTH
    return loss
if __name__ == "__main__":
    '''png = Image.open(r".\dataset2\darklabel\1020.tif")
    png=np.array(png)
    y=F1(png,png)
    print("F1",y)'''
   
    generate_arrays_from_file(418)