#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 21:13:08 2020

@author: josuenataren, aidanrogers, lydiaakino
"""

# first neural network with keras tutorial
from numpy import loadtxt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import csv
from matplotlib import pyplot
from numpy import mean, std


def keras_model():
    imgtrial = load_img('ImagesForKerasMod/DataPicture'+str(1)+'.png',
                        color_mode="rgb")
    img_arraytrial = img_to_array(imgtrial)

    # this variable is for the number of images for the training part
    numOfImages = 2000
    # this second variable is for the number of images for the testing part
    # of the model at the end
    numOfImages2 = 100
    # class types variable
    numOfTypes = 4

    # the data looks like the following:
    # x = [number_of_images, pixels]
    # y = [number_of_images, options]
    # the options for the classification are:   healthy      pT1       pT2
    # pT3

    x = np.zeros((numOfImages,
                  img_arraytrial.shape[0]*img_arraytrial.shape[1]))
    # print x.size
    # print 'initial x:'
    # print x

    print('Loading images for training:')

    for numImg in range(numOfImages):
        print(numImg)
        img = load_img('ImagesForKerasMod/DataPicture'
                       + str(numImg+1)+'.png', color_mode="rgb")
        # print(img.size)
        # show the image
        # img.show()
        # print("Orignal:" ,type(img))
        # convert to numpy array
        img_array = img_to_array(img)
        # print("NumPy array info:")
        # print(type(img_array))
        # print("type:",img_array.dtype)
        # print("shape:",img_array.shape)
        # #convert back to image

        imgValues = np.zeros((img_array.shape[0], img_array.shape[1]))

        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                imgValues[i][j] = (img_array[i][j][2] * 65536 +
                                   img_array[i][j][1] * 256 +
                                   img_array[i][j][0])/(256*256*256)

    # a = np.array([[1,2], [3,4]])
        imgValuesInput = np.resize(imgValues,
                                   (1, img_array.shape[0]*img_array.shape[1]))
    # imgValuesInput = imgValuesInput.T

    #    print imgValuesInput
        x[numImg][:] = imgValuesInput

    #
    # print 'updated x:'
    # print x

    # load the dataset
    # dataset = loadtxt('trialPics.csv')
    with open('trialPicsMod.csv', 'r') as f:
        reader = csv.reader(f)
        data_as_list = list(reader)

    y = np.zeros((numOfImages, numOfTypes))

    for i in range(numOfImages):
        for j in range(numOfTypes):
            y[i][j] = data_as_list[i][j]
    #
    # print 'this is y:'
    # print y

    print('x and y loaded successfully')
    # return img_array, x, y, numOfTypes, numOfImages2, img_arraytrial

# AT THIS POINT OF THE CODE: the images have been loaded and converted into
# arrays of decimals
# and stored in x as the input layer stuff: x(imageNumber, allThePoints)
# the variable y is the output variable to which the model will be fitted.
# Everything that needs to be worked on now is from here to the end of the
# code I think, to
# determine how many hidden layers we need, which type, and also how many
# images we need.


    # img_array, x, y, numOfTypes, numOfImages2, img_arraytrial = keras_setup()
    # define the keras model
    model = Sequential()
    model.add(Dense(1000, input_dim=img_array.shape[0]*img_array.shape[1],
                    activation='relu'))
    model.add(Dense(500, activation='relu'))
    #model.add(Dense(16, activation='relu'))
    # # model.add(Dense(28, activation='relu'))
    # model.add(Dense(numOfTypes, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy',
    #               optimizer='adam', metrics=['accuracy'])
    model.add(Dense(numOfTypes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # compile the keras model
    #


    # fit the keras model on the dataset
    history = model.fit(x, y, epochs=250, batch_size=500, verbose=1)
    # trains on 1,000 images 250 times, in sets of 15 images -->
    # 16,666.66667 iterations
    # trains on 1,000 images, 250 times, in sets of 100 --> 2,500 iterations

    # evaluate the keras model
    _, accuracy = model.evaluate(x, y, verbose=1)
    print('Accuracy: %.2f' % (accuracy*100))  # 99.76
    print('END OF TRAINING')
    accuracy_total, histories = list(), list()
    accuracy_total.append(accuracy)
    histories.append(history)
    # make probability predictions with the model
    predictions = model.predict(x)
    accuracy_performance(accuracy_total)


    # summarize the first 50 cases of TRAINING
    for i in range(50):
        print('actual from model (TRAINING image %s): %s => (exp. %s)'
              % (str(i+1), str(np.round(predictions[i][:])), str(y[i][:])))

    # checking for new images
    x2 = np.zeros((numOfImages2,
                   img_arraytrial.shape[0]*img_arraytrial.shape[1]))

    for numImg2 in range(numOfImages2):
        print(numImg2)
        img = load_img('ImagesForKerasMod/DataPictureTest'
                       + str(numImg2+1)+'.png', color_mode="rgb")
        # print(img.size)
        # show the image
        # img.show()
        # print("Orignal:" ,type(img))
        # convert to numpy array
        img_array = img_to_array(img)
        # print("NumPy array info:")
        # print(type(img_array))
        # print("type:",img_array.dtype)
        # print("shape:",img_array.shape)
        # #convert back to image

        imgValues = np.zeros((img_array.shape[0], img_array.shape[1]))

        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                imgValues[i][j] = (img_array[i][j][2] * 65536 +
                                   img_array[i][j][1] * 256
                                   + img_array[i][j][0])/(256*256*256)

    # a = np.array([[1,2], [3,4]])
        imgValuesInput = np.resize(imgValues,
                                   (1, img_array.shape[0]*img_array.shape[1]))
    # imgValuesInput = imgValuesInput.T

    #    print imgValuesInput
        x2[numImg2][:] = imgValuesInput

    # load the dataset
    with open('trialPicsMod2.csv', 'r') as f:
        reader = csv.reader(f)
        data_as_list2 = list(reader)

    y2 = np.zeros((numOfImages2, numOfTypes))

    for i in range(numOfImages2):
        for j in range(numOfTypes):
            y2[i][j] = data_as_list2[i][j]

    # make probability predictions with the model
    predictions2 = model.predict(x2)
    print('STUFF WITH NEW IMAGES:')
    # summarize the first 5 cases
    predictions_array, expected_array = list(), list()
    for i in range(numOfImages2):
        print('actual from model (NEW image %s): %s => (exp. %s)'
              % (str(i+1), str(np.round(predictions2[i][:])), str(y2[i][:])))

    training_performance(histories)

    numCorrect = 0
    numCorrectPerImg = 0

    for i in range(numOfImages2):
        dummyV = np.round(predictions2[i][:])
        dummyV2 = y2[i][:]
        for j in range(numOfTypes):
            if dummyV[j] == dummyV2[j]:
                numCorrectPerImg = numCorrectPerImg + 1
        if numCorrectPerImg == numOfTypes:
            numCorrect = numCorrect + 1
        numCorrectPerImg = 0
    AccuracyFinal = (numCorrect / numOfImages2) * 100
    print(AccuracyFinal)


def training_performance(histories):
    for i in range(len(histories)):
        # plot loss
        #
        pyplot.figure(2)
        pyplot.subplot(2, 1, 1)
        pyplot.title('Cross Entropy Loss')
        pyplot.xlabel('Epochs')
        pyplot.plot(histories[i].history['loss'],
                    color='blue', label='train')
        # pyplot.plot(histories[i].history['value_loss'],
        #             color='orange', label='test')
        # plot accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.title('Classification Accuracy')
        pyplot.xlabel('Epochs')
        pyplot.plot(histories[i].history['accuracy'],
                    color='blue', label='train')
        # pyplot.plot(histories[i].history['val_accuracy'],
        #             color='orange', label='test')
    pyplot.show()


# summarize model performance
def accuracy_performance(acc):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d'
          % (mean(acc)*100, std(acc)*100, len(acc)))


if __name__ == '__main__':
    keras_model()
