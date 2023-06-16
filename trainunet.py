from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from PIL import Image
from tensorflow.python.ops import array_ops
import time
import tensorflow.keras
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from model._model_unet_plus_2d import unet_plus_2d as create_model

import os
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

def getlines(path):
    imgs = os.listdir(path)
    lines=[]
    for jpg in imgs:
        line=path+'\\'+jpg+';'+path[0:-4]+'label\\'+jpg+'\n'
        lines.append(line)
        
    return lines  

def generate_arrays_from_file(lines,batch_size):
    
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
      
        for _ in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            name1 = (lines[i].split(';')[1]).replace("\n", "")
            

            
            #jpg = Image.open(r"..\样本集\训练集\img" + '/' + name)
            #png = Image.open(r"..\样本集\训练集\label" + '/' + name1)
            jpg = Image.open(name)
            png = Image.open(name1)
            #png,_,_ = letterbox_image(png,(HEIGHT,WIDTH),"png")
            jpg= jpg.resize((WIDTH,HEIGHT))
            png = png.resize([int(HEIGHT),int(WIDTH)])
            #jpg, png = get_random_data(jpg,png,[WIDTH,HEIGHT])
            #jpg,_,_ = letterbox_image(jpg,(WIDTH,HEIGHT),"jpg")
            
            jpg = np.array(jpg)
            jpg = jpg/255
          
            
            
            # ���ļ��ж�ȡͼ��
            

            
            png = np.array(png)
            
            seg_labels = np.zeros((int(HEIGHT),int(WIDTH),NCLASSES))
            for c in range(NCLASSES):
                seg_labels[:,:,c] = (png[:,:] == c ).astype(int)
            seg_labels = np.reshape(seg_labels, (-1,NCLASSES))

            X_train.append(jpg)
            Y_train.append(seg_labels)

            # ����һ�����ں����¿�ʼ
            i = (i+1) % n
            
        
        yield (np.array(X_train),np.array(Y_train))
@tf.function
def Recall(y_true,y_pred):
    true_pos=K.sum(K.round(y_pred)*y_true)
    possible_pos=K.sum(y_true)
    return true_pos/(possible_pos+K.epsilon())
@tf.function
def Precision(y_true,y_pred):
    true_pos=K.sum(K.round(y_pred)*y_true)
    pre_pos=K.sum(y_pred)
    return true_pos/(pre_pos+K.epsilon())
@tf.function
def F1(y_true,y_pred):
    pre=Precision(y_true,y_pred)
    recall=Recall(y_true,y_pred)
    return 2*(pre*recall)/(pre+recall+K.epsilon())
def Diceloss(y_true,y_pred):
    f1=F1(y_true,y_pred)
    return (1-f1)
@tf.function
def focalloss(ttensor,pretensor,alpha=0.25,gamma=2):
    zeros= array_ops.zeros_like(pretensor,dtype=pretensor.dtype)
    ttensor=tf.cast(ttensor,pretensor.dtype)
    pos=array_ops.where(ttensor>zeros,ttensor-pretensor,zeros)
    neg=array_ops.where(ttensor>zeros,zeros,pretensor)
    per_crossent=-alpha*(pos**gamma)*tf.math.log(tf.clip_by_value(1.0-pretensor,1e-8,1.0))\
                    -(1-alpha)*(neg**gamma)*tf.math.log(tf.clip_by_value(1.0-pretensor,1e-8,1.0))/HEIGHT/WIDTH
    return tf.math.reduce_sum(per_crossent)
@tf.function
def TverskyLoss(y_true,y_pred,smooth,alpha,beta):
    tp=K.sum(K.round(y_pred)*y_true)
    fp=[]
    tversky=[]
@tf.function
def loss(y_true, y_pred):
    crossloss = K.categorical_crossentropy(y_true,y_pred)
    loss = K.sum(crossloss)/HEIGHT/WIDTH
    return loss
if __name__ == "__main__":
    log_dir = "logs/"
    # ��ȡmodel
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    with strategy.scope():
        #tf.keras.backend.set_floatx('float32')
    #model = Deeplabv3(input_shape=(HEIGHT,WIDTH,NCLASSES),classes=2)
        model = create_model((512,512,3),2)
        model.summary()

        #weights_path = r".\logs\middle1.h5"
        #print(weights_path)
        #model.load_weights(weights_path)
    #   model.load_weights(weights_path,by_name=True)
    # �����ݼ���txt
    #with open(r"..\样本集\训练集\train_data.txt","r") as f:
        #lines = f.readlines()

    # �����У����txt��Ҫ���ڰ�����ȡ������ѵ��
    # ���ҵ����ݸ�������ѵ��
    
    '''lines0=getlines('E:\海漂垃圾自动识别\样本集\全部垃圾提取\惠安县山霞镇东埭村岸段img')
    lines1=getlines('E:\海漂垃圾自动识别\样本集\全部垃圾提取\蕉城区漳湾镇鳌江岸段img')
    lines2=getlines('E:\海漂垃圾自动识别\样本集\全部垃圾提取\霞浦溪南镇南坂至仙东岸段img')
    lines3=getlines('E:\海漂垃圾自动识别\样本集\全部垃圾提取\霞浦县下浒镇前洋至西岐岸段img')
    lines4=getlines('E:\海漂垃圾自动识别\样本集\全部垃圾提取\秀屿区东埔镇塔林岸段img')
    lines=lines0+lines1+lines2+lines3+lines4'''
    lines=getlines("E:\海漂垃圾自动识别\样本集\废弃撑架\positive\img2")
    #print(lines)
    '''nlines=getlines('F:\\海漂垃圾自动识别\\样本集\\训练集\\negtive')
    np.random.shuffle(nlines)
    nnlines=nlines[1:1000]
    for line in nlines:
        lines.append(line)'''
    


 
    np.random.seed(5037)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%����ѵ����10%���ڹ��ơ�
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val

    # ����ķ�ʽ��3��������һ�� 
    checkpoint_period = ModelCheckpoint(
                                    log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_loss', 
                                    save_weights_only=True, 
                                    save_best_only=True, 
                                    period=1
                                )
                                
    '''mc = ModelCheckpoint(mode='min', filepath='top_weights.h5',
                     monitor='val_loss',
                     save_best_only='True',
                     save_weights_only='True', verbose=1)   '''                     
    # ѧϰ���½��ķ�ʽ��val_loss 2�β��½����½�ѧϰ�ʼ���ѵ��
    reduce_lr = ReduceLROnPlateau(
                            monitor='val_loss', 
                            factor=0.5, 
                            patience=3, 
                            verbose=1
                        )
    # �Ƿ���Ҫ��ͣ����val_lossһֱ���½���ʱ����ζ��ģ�ͻ���ѵ����ϣ�����ֹͣ
    early_stopping = EarlyStopping(
                            monitor='val_loss', 
                            min_delta=0, 
                            patience=6, 
                            verbose=1
                        )
    # ������
    #optimizer = RMSprop(learning_rate=1e-4,momentum=0.01),
    #optimizer = Adam(learning_rate=1e-4),
    with strategy.scope():
        model.compile(loss = focalloss,
            optimizer = Adam(learning_rate=1e-3),
            metrics = [[Precision],[Recall],[F1]])
    batch_size = 6
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    # ��ʼѵ��
    model.fit(generate_arrays_from_file(lines[:num_train], batch_size),
            steps_per_epoch=max(1, num_train//batch_size),
            epochs=200,
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
            validation_steps=max(1, num_val//batch_size),
            callbacks=[reduce_lr,checkpoint_period]
            )
    '''model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
            validation_steps=max(1, num_val//batch_size),
            epochs=30,
            initial_epoch=0,
            callbacks=[checkpoint_period, reduce_lr, early_stopping]
            )'''

    model.save_weights(log_dir+'middle1.h5')

   #������
    '''model.compile(loss = loss,
            optimizer = Adam(lr=1e-4),
            metrics = ['accuracy'])
    batch_size = 1
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    # ��ʼѵ��
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
            validation_steps=max(1, num_val//batch_size),
            epochs=60,
            initial_epoch=30,
            callbacks=[checkpoint_period, reduce_lr, early_stopping])

    model.save_weights(log_dir+'last1.h5')'''