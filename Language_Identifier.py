import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
import sys

#This file will be more of a script file that prints results to terminal

#This file will import the other files such as the image_identifier model and pass on that to a Langauge detection model
#Will be testing with the CNN and RF models since those should be the most accurate ones for the Image Classification



# Give the folder pathway in order to locate the image_to_text.py or else will go through entire system until first match up with the file name


sys.path.append('..../handwritten-language-detection')
from image_to_text import Word

#In this file the first input is the handwritten image, second is the model name, and the third id hte model label name include extensions in all three
#If using the CNN, do not include extensions in the second input



#Enter pathway leading up to this point
#Example of models being used
#For path_model, if using CNN do not enter the extension such as .pickle or .h5

path_image = '......./handwritten-language-detection/analysis/images/Project_Image4.jpg'

path_model = '....../handwritten-language-detection/CNN/CNNadamaxE6_model'

path_model_label = '......./handwritten-language-detection/CNN/CNNadamaxE6_labels.pickle'




word = Word(path_image, path_model, path_model_label)
word = word.driver()

#w will contain the classified text word from the handwritten image
#converts it to a nuemrical value in order to be accesed by the LNaguage detection model
#path to language vectorizer


#Example of a model that is compiled
path_vec = '............/handwritten-language-detection/KNN_Language/KNN3vectorizer.pickle'

with open(path_vec, 'rb') as f:
    vector = pickle.load(f)



word = word.lower()
word = word.replace(" ", "") #removes the whitespaces in between the word


text_word = vector.transform([word])
text_word.toarray()



language_model_path = '............/handwritten-language-detection/KNN_Language/KNN3_model.pickle'
#This opens the trained model
with open(language_model_path, 'rb') as f:
    model = pickle.load(f)


pred = model.predict(text_word)


encoder_model_path = '............/handwritten-language-detection/KNN_Language/KNN3labelencoder.pickle'
with open(encoder_model_path, 'rb') as f:
    le = pickle.load(f)


pred = le.inverse_transform(pred)

#This prints the Final Prediction to terminal
print(word, 'is written in', pred[0])
