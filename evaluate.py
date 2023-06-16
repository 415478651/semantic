from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from PIL import Image
import time
import tensorflow.keras
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from modelcopy2copy import efficientnetv2_s as create_model

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


def generate_arrays_from_file(lines,batch_size):
    # ��ȡ�ܳ���
    n = len(lines)
    i = 0

    X_train = []
    Y_train = []
    # ��ȡһ��batch_size��С������
    while 1:
        for _ in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            name1 = (lines[i].split(';')[1]).replace("\n", "")
            # ���ļ��ж�ȡͼ��
            jpg = Image.open(r".\dataset2\image" + '/' + name)
            png = Image.open(r".\dataset2\label" + '/' + name1)
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
def Recall(y_true,y_pred):
    true_pos=K.sum(K.round(y_pred)*y_true)
    possible_pos=K.sum(y_true)
    return true_pos/(possible_pos+K.epsilon())
def Precision(y_true,y_pred):
    true_pos=K.sum(K.round(y_pred)*y_true)
    pre_pos=K.sum(y_pred)
    return true_pos/(pre_pos+K.epsilon())
def F1(y_true,y_pred):
    pre=Precision(y_true,y_pred)
    recall=Recall(y_true,y_pred)
    return 2*(pre*recall)/(pre+recall+K.epsilon())
def Diceloss(y_true,y_pred):
    f1=F1(y_true,y_pred)
    return (1-f1)
def loss(y_true, y_pred):
    crossloss = K.categorical_crossentropy(y_true,y_pred)
    loss = K.sum(crossloss)/HEIGHT/WIDTH
    return loss
if __name__ == "__main__":
    log_dir = "logs/"
    # ��ȡmodel
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        #tf.keras.backend.set_floatx('float32')
    #model = Deeplabv3(input_shape=(HEIGHT,WIDTH,NCLASSES),classes=2)
        model = create_model(num_classes=NCLASSES,img_height=HEIGHT,img_width=WIDTH)
        model.summary()

        weights_path = r"F:\tensorflow_classification\efficientnetV2\logs\middle1.h5"
        print(weights_path)
        model.load_weights(weights_path)
    #   model.load_weights(weights_path,by_name=True)
    # �����ݼ���txt
    with open(r".\dataset2\test.txt","r") as f:
        lines = f.readlines()

   
    #optimizer = RMSprop(learning_rate=1e-4,momentum=0.01),
    #optimizer = Adam(learning_rate=1e-4),
    with strategy.scope():
        model.compile(loss = loss,
            metrics = ['accuracy',Precision,Recall,F1])
    batch_size = 1
    #img,label=generate_arrays_from_file(lines,batch_size)
    n = len(lines)
    n = len(lines)
    i = 0
    losssum=0
    accsum=0
    presum=0
    recsum=0
    f1sum=0
    X_train = []
    Y_train = []
    # ��ȡһ��batch_size��С������
    for _ in range(1,int(n/batch_size)):
        for _ in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            name1 = (lines[i].split(';')[1]).replace("\n", "")
            # ���ļ��ж�ȡͼ��
            jpg = Image.open(r".\dataset2\image" + '/' + name)
            png = Image.open(r".\dataset2\label" + '/' + name1)
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
            
        loss,acc,pre,rec,f1=model.evaluate(np.array(X_train),np.array(Y_train))
        
        
        losssum=losssum+loss
        accsum=accsum+acc
        presum=presum+pre
        recsum=recsum+rec
        f1sum=f1sum+f1
        print(i)
    losssum=losssum/202
    accsum=accsum/202
    presum=presum/202
    recsum=recsum/202
    f1sum=f1sum/202
    print(loss,acc,pre,rec,f1)
    #result=model.evaluate(generate_arrays_from_file(lines, batch_size))
    print(result)
    '''model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
            validation_steps=max(1, num_val//batch_size),
            epochs=30,
            initial_epoch=0,
            callbacks=[checkpoint_period, reduce_lr, early_stopping]
            )'''

    #model.save_weights(log_dir+'middle1.h5')

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