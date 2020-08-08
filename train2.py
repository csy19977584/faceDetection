import random

import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from load_face_data import load_dataset, resize_image, IMAGE_SIZE
from centerloss import loss, categorical_accuracy
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
from losses import center_loss
from keras import metrics


class Dataset:
    def __init__(self, path_name1, path_name2):

        self.train_images = None
        self.train_labels = None

        self.valid_images = None
        self.valid_labels = None

        self.test_images = None
        self.test_labels = None

        self.path_name1 = path_name1
        self.path_name2 = path_name2

        self.input_shape = None

    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE,img_channels=3, nb_classes = 5):
        images, labels = load_dataset(self.path_name1)
        test_images, test_labels = load_dataset(self.path_name2)

        train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size=0.3,random_state=random.randint(0, 100))
        #改为none试试，改参数试试

        if K.image_dim_ordering() == 'th':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)

            print(train_images.shape, 'train samples')
            print(valid_images.shape, 'valid samples')
            print(test_images.shape, 'test samples')

            train_labels = np_utils.to_categorical(train_labels, nb_classes)
            valid_labels = np_utils.to_categorical(valid_labels, nb_classes)
            test_labels = np_utils.to_categorical(test_labels, nb_classes)

            train_images = train_images.astype('float32')
            valid_images = valid_images.astype('float32')
            test_images = test_images.astype('float32')

            train_images /= 255
            valid_images /= 255
            test_images /= 255
            #数据预处理对于网络的性能也有一定影响 举例，图像文件的原始像素往往位于
            #[O, 255 ］区间，如果直接将这样大的数字输入网络，很容易造成不稳定 我们可将其初始化到
            #[O, ］或［－ 1, 间，或将其处理为均值为 、标准差为 0

            self.train_images = train_images
            self.valid_images = valid_images
            self.test_images = test_images
            self.train_labels = train_labels
            self.valid_labels = valid_labels
            self.test_labels = test_labels

def generateData(batch_size,imgs,labels):
    #print 'generateData...'
    while True:
        train_data1 = []
        train_data2 = []

        batch = 0
        for i in (range(len(imgs))):
            img=imgs[i]
            label=labels[i]
            batch += 1
            train_data1.append(img)
            train_data2.append(label)

            if batch % batch_size==0:
                #print 'get enough bacth!\n'
                train_data1 = np.array(train_data1)
                train_data2 = np.array(train_data2)
                yield (train_data1,[train_data2,train_data2])
                train_data1 = []
                train_data2 = []
                batch = 0

def dataEnhancement(generator, batch_size, imgs, labels):
    genX = generator.flow(imgs, labels, batch_size)
    while True:
        Xi = genX.next()
        print(1)
        yield Xi[0], [Xi[1], Xi[1]]


class Model:
    def __init__(self):
        self.model = None

    def build_model(self, dataset, nb_classes):
        input = Input(dataset.input_shape)
        x = Conv2D(32, (3, 3), activation='relu')(input)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = Dropout(0.25)(x)#qudiao
        x = Flatten()(x)

        feature = Dense(256, activation='relu',name='out1')(x)#256--50
        x = Dropout(0.5)(feature)
        out = Dense(nb_classes,activation='softmax',name='out2')(x)
        self.model = keras.models.Model(inputs=input, outputs=[feature,out])

    def train(self, dataset, batch_size=20, nb_epoch=10, data_augmentation=True):

        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        #sgd=Adam(lr=0.001)
        self.model.compile(loss={'out1':center_loss,'out2':'categorical_crossentropy'},
                           loss_weights=[0.00001,1],#0.0001，0.00001
                           optimizer=sgd,
                           metrics={'out2':'accuracy'})
        # metrics = {'out': 'categorical_accuracy'})
        # 初始化变量
        sess = K.get_session()#从Keras模型定义中获取输入和输出张量
        sess.run(tf.global_variables_initializer())
        print("test")
        if not data_augmentation:
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size=batch_size,
                           epochs=nb_epoch,
                           shuffle=True)

        else:

            datagen = ImageDataGenerator(
                featurewise_center=False,  # 是否使输入数据去中心化（均值为0），
                samplewise_center=False,  # 是否使输入数据的每个样本均值为0
                featurewise_std_normalization=False,  # 是否数据标准化（输入数据除以数据集的标准差）
                samplewise_std_normalization=False,  # 是否将每个样本数据除以自身的标准差
                zca_whitening=False,  # 是否对输入数据施以ZCA白化
                rotation_range=20,  # 数据提升时图片随机转动的角度(范围为0～180)
                width_shift_range=0.2,  # 数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
                height_shift_range=0.2,  # 同上，只不过这里是垂直
                horizontal_flip=True,  # 是否进行随机水平翻转
                vertical_flip=False)

            #datagen.fit(dataset.train_images)

            self.model.fit_generator(generateData(batch_size,dataset.train_images,dataset.train_labels),
                                     steps_per_epoch=dataset.train_images.shape[0]//batch_size,
                                     epochs=nb_epoch,
                                     verbose=1,
                                     validation_steps=dataset.valid_images.shape[0]//batch_size,
                                     validation_data=generateData(batch_size,dataset.valid_images,dataset.valid_labels))

    MODEL_PATH = 'L:/dabian/model/dabian11.face.model.h5'

    def save_model(self, file_path=MODEL_PATH):
        self.model.save(file_path)

    def load_model(self, file_path=MODEL_PATH):
        self.model = load_model(file_path, custom_objects={'center_loss': center_loss})

    def evaluate(self, dataset):
        """
        score = self.model.evaluate(dataset.test_images, [dataset.test_labels,dataset.test_labels], verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
        """
        score = self.model.evaluate(dataset.test_images, [dataset.test_labels, dataset.test_labels], verbose=1)
        print(self.model.metrics_names)
        print('Test score:', score)

    def face_predict(self, image):
        if K.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))
        elif K.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))

        image = image.astype('float32')
        image /= 255

        _,result = self.model.predict(image)
        print("result",result)
        result = np.argmax(result, axis=1)
        return result


if __name__ == '__main__':
    dataset = Dataset('L:/saveimage1', 'L:/saveimage2')
    dataset.load(nb_classes=5)

    model = Model()
    model.build_model(dataset, nb_classes= 5)

    model.train(dataset)
    model.save_model(file_path='L:/dabian/model/dabian11.face.model.h5')
    print("the end")
    model.evaluate(dataset)


