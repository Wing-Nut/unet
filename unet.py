import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import Model,Input
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import array_to_img
import glob
from keras.models import load_model
from skimage.io import imread,imsave

class myUnet(object):
    def __init__(self,results_path):
        self.img_rows = None
        self.img_cols = None
        self.img_type = None

        try:
            self.model = load_model("/home/ubuntu/unet.hdf5")
        except:
            self.model = None

        self.results_path = results_path

    def load_training_data(self,mydata_object):
        self.img_cols = mydata_object.out_cols
        self.img_rows = mydata_object.out_rows

        imgs_train, imgs_mask_train = mydata_object.load_train_data()
        return imgs_train, imgs_mask_train

    def get_unet(self):

        inputs = Input((self.img_rows, self.img_cols,1))

        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        print("conv1 shape:",conv1.shape)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        print("conv1 shape:",conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print("pool1 shape:",pool1.shape)

        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        print("conv2 shape:",conv2.shape)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        print("conv2 shape:",conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print("pool2 shape:",pool2.shape)

        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        print("conv3 shape:",conv3.shape)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        print("conv3 shape:",conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print("pool3 shape:",pool3.shape)

        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

        model = Model(input = inputs, output = conv10)

        model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

        return model

    def train(self,myData,nb_epoch=5):
        # self.img_type = myData.img_type

        print("loading data")
        imgs_train, imgs_mask_train = self.load_training_data(myData)
        print("loading data done")
        self.model = self.get_unet()
        print("got unet")

        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
        print('Fitting model...')
        self.model.fit(imgs_train, imgs_mask_train, batch_size=4, nb_epoch=nb_epoch, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

        # self.model.save("/home/ubuntu/mymodel")



    def predict_and_save(self, mydata,my_set="test"):
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data()

        print('predict test data')
        print(imgs_train.shape)
        print(imgs_test.shape)
        if my_set == "test":
            imgs_mask = self.model.predict(imgs_test, batch_size=1, verbose=1)
            data_path = mydata.test_path
        else:
            imgs_mask = self.model.predict(imgs_train, batch_size=1, verbose=1)
            data_path = mydata.data_path

        # np.save(self.results_path +'/imgs_mask_'+my_set+'.npy', imgs_mask)

        print("array to image")
        # imgs = p.load('imgs_mask_test.npy')
        for i,full_path in enumerate(glob.glob(data_path  +"/*."+mydata.img_type)):
            img = imgs_mask[i]
            img = array_to_img(img)

            assert isinstance(full_path,str)
            # extract just the file name (not the path)
            f = full_path[full_path.rindex("/"):]

            img.save(self.results_path + f)

    # def predict_test_cases(self,mydata):
    #     mydata.create_test_data()
    #     test_images = mydata.load_test_data()
    #
    #
    #
    #     imgs_mask_test = self.model.predict(imgs_test, batch_size=1, verbose=1)
    #
    #     np.save('./results/imgs_mask_test.npy', imgs_mask_test)
    #
    #     print("array to image")
    #     imgs = np.load('./results/imgs_mask_test.npy')
    #     for i in range(imgs.shape[0]):
    #         img = imgs[i]
    #         img = array_to_img(img)
    #         img.save("./results/%d.png" % (i))


    def predict(self,image):
        assert self.model is not None

        out_rows,out_cols = image.shape[:2]

        # convert it into a format we are used to using
        imgdatas = np.ndarray((1, out_rows, out_cols, 1), dtype=np.uint8)

        imgdatas[0] = image.reshape((image.shape[0],image.shape[1],1))

        imgdatas = imgdatas.astype('float32')
        imgdatas /= 255

        mask = self.model.predict(imgdatas, batch_size=1, verbose=1)[0,:,:,:]

        return mask

if __name__ == '__main__':
    myunet = myUnet()
    myunet.train()
    myunet.save_img()








