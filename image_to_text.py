import cv2
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import imutils
from imutils.contours import sort_contours
import tensorflow as tf
from tensorflow.keras.models import model_from_json


# Majority of algorithm for getting characters from words into boudning boxes was obtained from tutorialpoints.com
# Manipulated to make it custom for my specific image types, 3 channels, add padding ,and etc
# This file takes in image, model, and model labels done from the other folders

class Word:

    def __init__(self, path_image, model, model_labels):    

        self.path = path_image
        self.model = model
        self.model_labels = model_labels

    def driver(self):   
        image = cv2.imread(self.path)

        #image will be from notability
        image = cv2.resize(image, (1200,1200))

        #For analysis purposes
        plt.imshow(image)
        plt.show()

        #manipulated the contour/box for objects algo found at ...... 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)


        edged = cv2.Canny(blurred, 30, 150)


        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)


        cnts = imutils.grab_contours(cnts)
        cnts = sort_contours(cnts, method="left-to-right")[0]


        chars = []
        box = []

        # loop over the contours, for getting the contours/bounding boxes the alogirthm was displayed on stackoverflow, had to manipoulate for different results in my case
        # Needed to deal with an rgb image for instance as well as add extra padding to fit my own dataset

        
        for c in cnts:

                (x, y, w, h) = cv2.boundingRect(c)

                if True:

                    roi = gray[y:y + h, x:x + w]
                    thresh = cv2.threshold(roi, 0, 255,
                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                    (tH, tW) = thresh.shape
                    # if the width is greater than the height, resize along the
                    # width dimension
                    if tW > tH:
                        thresh = imutils.resize(thresh, width=100)
                    # otherwise, resize along the height
                    else:
                        thresh = imutils.resize(thresh, height=100)

                    (tH, tW) = thresh.shape
                    dX = int(max(0, 100 - tW) / 2.0)
                    dY = int(max(0, 100 - tH) / 2.0)
                    # pad the image and force 32x32 dimensions
                    padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                        left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                        value=(0, 0, 0))

                    padded = cv2.resize(padded, (100, 100))
                    # prepare the padded image for classification via our
                    # handwriting OCR model
                    padded = padded.astype("float32") / 255.0
                    #padded = cv2.normalize(padded, None, 0, 1, cv2.NORM_MINMAX)
                    padded = np.expand_dims(padded, axis=-1)
                    padded = 1 - padded

                    
                    padded = cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)

                    plt.imshow(padded)
                    plt.show()

                    # update our list of characters that will be OCR'd
                    chars.append(padded)
                    box.append(x)



        order_padding =  []
        for i in range(len(chars)):
            index = box.index(min(box))
            order_padding.append(chars[index])
            chars.remove(chars[index])
            box.remove(min(box))


        for i in range(len(order_padding)):
            ######################################################
            #Adding white pixels around to center the letter so it resembles images in the dataset or else wont be as accurate
            WHITE = [1,1,1]
            order_padding[i] = cv2.copyMakeBorder(order_padding[i],60,60,60,60,cv2.BORDER_CONSTANT,value=WHITE)
            order_padding[i] = cv2.resize(order_padding[i], (100,100))
            ######################################################   

        for i in range(len(order_padding)):
            plt.imshow(order_padding[i])
            plt.show()

        word = []


        #This opens the lables that were saved
        with open(self.model_labels, 'rb') as f:
            labels = pickle.load(f)



        #Depending on the method that is used will have to have different ways of opening up label

        #To know if you are using the CNN folder
        if 'CNN' in self.model:

            # load json and create model
            json_file = open(self.model + '.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights(self.model +'.h5')

            
            # evaluate loaded model on test data

            if 'CNNadamax' in self.model:

                loaded_model.compile(optimizer='adamax',
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'])
            
            else:

                loaded_model.compile(optimizer='adam',
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'])
            

            
            for i in order_padding:

                img = i
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0) 
                predictions = loaded_model.predict(img_array)
                classes = np.argmax(predictions, axis = 1)
                keys = list(labels.keys())
                vals = list(labels.values())
                word.append(str(keys[vals.index(classes)]))

            concat_word = ' '.join(word) 


        else:
            #This opens the trained model
            with open(self.model, 'rb') as f:
                model = pickle.load(f)

            for i in order_padding:
                img = i
                img = img.ravel()
                img = [img]
                predictions = model.predict(img)
                classes = predictions[0]
                keys = list(labels.keys())
                vals = list(labels.values())
                word.append(str(keys[vals.index(classes)]))


            concat_word = "".join(word)


        return concat_word


# #Enter pathway leading up to this point
# #Example of models being used
# #For path_model, if using CNN do not enter the extension such as .pickle or .h5

# path_image = '......./analysis/images/Project_Image4.jpg'

# path_model = '....../CNN/CNNadamaxE6_model'

# path_model_label - '......./CNN/CNNadamaxE6_labels.pickle'


# word = Word(path_image, path_model, path_model_label)
# text_word = word.driver()
