import os
import math
import datetime

import tensorflow as tf
from tqdm import tqdm

from model import efficientnetv2_s as create_model


#assert tf.version.VERSION >= "2.4.0", "version of tf must greater/equal than 2.4.0"

def generate_arrays_from_file(lines,batch_size):
    # ��ȡ�ܳ���
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # ��ȡһ��batch_size��С������
        for _ in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # ���ļ��ж�ȡͼ��
            jpg = Image.open(r"..\样本集\训练集\img" + '/' + name)
            #jpg,_,_ = letterbox_image(jpg,(WIDTH,HEIGHT),"jpg")
            jpg= jpg.resize((WIDTH,HEIGHT))
            jpg = np.array(jpg)
            jpg = jpg/255
          
            
            name = (lines[i].split(';')[1]).replace("\n", "")
            # ���ļ��ж�ȡͼ��
            png = Image.open(r"..\样本集\训练集\label" + '/' + name)
            #png,_,_ = letterbox_image(png,(HEIGHT,WIDTH),"png")

            #jpg, png = get_random_data(jpg,png,[WIDTH,HEIGHT])

            png = png.resize([int(HEIGHT),int(WIDTH)])
            png = np.array(png)
            seg_labels = np.zeros((int(HEIGHT),int(WIDTH),NCLASSES))
            for c in range(NCLASSES):
                seg_labels[: , : , c ] = (png[:,:] == c ).astype(int)
            seg_labels = np.reshape(seg_labels, (-1,NCLASSES))

            X_train.append(jpg)
            Y_train.append(seg_labels)

            # ����һ�����ں����¿�ʼ
            i = (i+1) % n
        yield (np.array(X_train),np.array(Y_train))
def main():
    #data_root = "/data/flower_photos"  # get data root path

    if not os.path.exists("./save_weights"):
        os.makedirs("./save_weights")
    #"s": [300, 384]
    img_size = {"s": [512, 512],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "s"

    batch_size = 8
    epochs = 30
    num_classes = 2
    freeze_layers = True
    initial_lr = 0.01

    log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
    val_writer = tf.summary.create_file_writer(os.path.join(log_dir, "val"))
    with open(r".\dataset2\train_data_shuitou.txt","r") as f:
        lines = f.readlines()
    # data generator with data augmentation
    train_ds, val_ds = generate_arrays_from_files()

    # create model
    model = create_model(num_classes=num_classes)

    #pre_weights_path = './efficientnetv2-s.h5'
    #assert os.path.exists(pre_weights_path), "cannot find {}".format(pre_weights_path)
    #model.load_weights(pre_weights_path, by_name=True, skip_mismatch=True)

    # freeze bottom layers
    '''if freeze_layers:
        unfreeze_layers = "head"
        for layer in model.layers:
            if unfreeze_layers not in layer.name:
                layer.trainable = False
            else:
                print("training {}".format(layer.name))'''

    model.summary()

    # custom learning rate curve
    def scheduler(now_epoch):
        end_lr_rate = 0.01  # end_lr = initial_lr * end_lr_rate
        rate = ((1 + math.cos(now_epoch * math.pi / epochs)) / 2) * (1 - end_lr_rate) + end_lr_rate  # cosine
        new_lr = rate * initial_lr

        # writing lr into tensorboard
        with train_writer.as_default():
            tf.summary.scalar('learning rate', data=new_lr, step=epoch)

        return new_lr

    # using keras low level api for training
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=initial_lr, momentum=0.9)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    @tf.function
    def train_step(train_images, train_labels):
        with tf.GradientTape() as tape:
            output = model(train_images, training=True)
            loss = loss_object(train_labels, output)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(train_labels, output)

    @tf.function
    def val_step(val_images, val_labels):
        output = model(val_images, training=False)
        loss = loss_object(val_labels, output)

        val_loss(loss)
        val_accuracy(val_labels, output)

    best_val_acc = 0.
    for epoch in range(epochs):
        train_loss.reset_states()  # clear history info
        train_accuracy.reset_states()  # clear history info
        val_loss.reset_states()  # clear history info
        val_accuracy.reset_states()  # clear history info

        # train
        train_bar = tqdm(train_ds)
        for images, labels in train_bar:
            train_step(images, labels)

            # print train process
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                                 epochs,
                                                                                 train_loss.result(),
                                                                                 train_accuracy.result())

        # update learning rate
        optimizer.learning_rate = scheduler(epoch)

        # validate
        val_bar = tqdm(val_ds)
        for images, labels in val_bar:
            val_step(images, labels)

            # print val process
            val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                               epochs,
                                                                               val_loss.result(),
                                                                               val_accuracy.result())
        # writing training loss and acc
        with train_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), epoch)
            tf.summary.scalar("accuracy", train_accuracy.result(), epoch)

        # writing validation loss and acc
        with val_writer.as_default():
            tf.summary.scalar("loss", val_loss.result(), epoch)
            tf.summary.scalar("accuracy", val_accuracy.result(), epoch)

        # only save best weights
        if val_accuracy.result() > best_val_acc:
            best_val_acc = val_accuracy.result()
            save_name = "./save_weights/efficientnetv2.ckpt"
            model.save_weights(save_name, save_format="tf")


if __name__ == '__main__':
    main()
