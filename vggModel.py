import tensorflow as tf
import cv2
import numpy as np
import os

def x_train_generator(photo_num=1, photo_size=(224, 224)):
    for i in range(photo_num):
        img = cv2.imread(os.path.join(os.getcwd(), "dog", f"{i}.jpeg"))
        img = cv2.resize(img, photo_size)
        img = img/255
        yield img

        
def mk_model():
    input = tf.keras.layers.Input(shape=(224, 224, 3))
    vgg_block1 = tf.keras.layers.Conv2D(64, (3,3), padding = "SAME", activation="relu")(input)
    vgg_block1 = tf.keras.layers.Conv2D(64, (3,3), padding = "SAME", activation="relu")(vgg_block1)
    vgg_block1 = tf.keras.layers.MaxPool2D((2, 2))(vgg_block1)  #112

    vgg_block2 = tf.keras.layers.Conv2D(128, (3,3), padding = "SAME", activation="relu")(vgg_block1)
    vgg_block2 = tf.keras.layers.Conv2D(128, (3,3), padding = "SAME", activation="relu")(vgg_block2)
    vgg_block2 = tf.keras.layers.MaxPool2D((2, 2))(vgg_block2) #56

    vgg_block3 = tf.keras.layers.Conv2D(256, (3,3), padding = "SAME", activation="relu")(vgg_block2)
    vgg_block3 = tf.keras.layers.Conv2D(256, (3,3), padding = "SAME", activation="relu")(vgg_block3)
    vgg_block3 = tf.keras.layers.Conv2D(256, (3,3), padding = "SAME", activation="relu")(vgg_block3)
    vgg_block3 = tf.keras.layers.MaxPool2D((2, 2))(vgg_block3) #28

    vgg_block4 = tf.keras.layers.Conv2D(512, (3,3), padding = "SAME", activation="relu")(vgg_block3)
    vgg_block4 = tf.keras.layers.Conv2D(512, (3,3), padding = "SAME", activation="relu")(vgg_block4)
    vgg_block4 = tf.keras.layers.Conv2D(512, (3,3), padding = "SAME", activation="relu")(vgg_block4)
    vgg_block4 = tf.keras.layers.MaxPool2D((2, 2))(vgg_block4) #14

    vgg_block5 = tf.keras.layers.Conv2D(512, (3,3), padding = "SAME", activation="relu")(vgg_block4)
    vgg_block5 = tf.keras.layers.Conv2D(512, (3,3), padding = "SAME", activation="relu")(vgg_block5)
    vgg_block5 = tf.keras.layers.Conv2D(512, (3,3), padding = "SAME", activation="relu")(vgg_block5)
    vgg_block5 = tf.keras.layers.MaxPool2D((2, 2))(vgg_block5) #7

    flatten = tf.keras.layers.Flatten()(vgg_block5)
    dense1 = tf.keras.layers.Dense(4096, activation = "relu")(flatten)
    output1 = tf.keras.layers.Dense(1000, activation = "softmax")(dense1)
    output2 = tf.keras.layers.Dense(1, activation = "sigmoid")(output1)

    model = tf.keras.models.Model(input, output2)
    return model

if __name__=="__main__":
    dog_num = 9
    cat_num = 9
    x_train = np.array(list(x_train_generator(photo_num = dog_num+cat_num)))
    y_train = np.concatenate([np.zeros((9, 1)), np.ones((9, 1))], axis=0)
    print(x_train.shape)
    print(y_train.shape)
    x_test = cv2.imread("./dog/0.jpeg")
    x_test = cv2.resize(x_test, (224, 224))
    x_test = np.expand_dims(x_test, axis=0)
    x_test = x_test/255
    
    model = mk_model()
    model.compile(optimizer = "adam", loss="binary_crossentropy", metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=64 )
    rate = model.predict(x_test)
    
    print(f"고양이일 확률= {rate[0][0]} ")  
    if rate[0][0] >= 0.5:
        x_test = x_test[0, :, :, :]
        cv2.imshow("Cat", x_test)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        x_test = x_test[0, :, :, :]
        cv2.imshow("Dog", x_test)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


