import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pickle



class KNN_Language:

    def __init__(self, path):

        self.path = path

    def driver(self):

        data = pd.read_csv(self.path)

        #Sets the conditions for the Latin Alphabet languages that I want to use.
        data = data.loc[(data['Language'] == 'English') | (data['Language'] == 'Spanish') | (data['Language'] == 'French') | (data['Language'] == 'Portugeese') | (data['Language'] == 'Italian') | (data['Language'] == 'Sweedish') | (data['Language'] == 'Dutch') | (data['Language'] == 'Danish') | (data['Language'] == 'German')]


        # To clean the data have to remove any possible symbols and number

        #Removes any unwanted characters: Doing most common as well as from a bit of data parsing
        #This way also helps visualixe which characters might be missing
        data['Text'] = data['Text'].str.replace('\d+', '')
        data['Text'] = data['Text'].str.replace('"' , '') 
        data['Text'] = data['Text'].str.replace(',', '') 
        data['Text'] = data['Text'].str.replace('[', '')
        data['Text'] = data['Text'].str.replace(']', '')
        data['Text'] = data['Text'].str.replace('.', '')
        data['Text'] = data['Text'].str.replace('?', '')
        data['Text'] = data['Text'].str.replace(';', '')
        data['Text'] = data['Text'].str.replace('\'', '')
        data['Text'] = data['Text'].str.replace('-', '')
        data['Text'] = data['Text'].str.replace('\\', '')
        data['Text'] = data['Text'].str.replace('%', '')
        data['Text'] = data['Text'].str.replace('(', '')
        data['Text'] = data['Text'].str.replace(')', '')


        #Lower case all the words
        data['Text'] = data['Text'].str.lower()

        #Now need to encode the data, so will convert fromn categorical to numerical. This is pretty simple, basically becomes key-value 

        data['Language'] = data['Language'].astype('category')
        data['Language Numerical'] = data['Language'].cat.codes


        #this will be alled upon for later usage
        #Used below for separation of sentences to words
        le = LabelEncoder()

        #Can use this as a reference dictionary
        lang_dict = {'Danish': 0,
            'Dutch': 1,
            'English': 2,
            'French': 3,
            'German': 4,
            'Italian': 5,
            'Portugeese': 6,
            'Spanish': 7,
            'Sweedish': 8}


        #Lang Numerical is the numerical value for each Language. Look at the above dictionary for the values
        list_label = []
        for j in data['Language Numerical']:
            list_label.append(j)


        #Organizes the initial data (sentences) into just words. List for words, language, and numerical value of language is done below.
        word = []
        word_label = []
        word_language = []


        counter = 0
        for sentence in data['Text']:
            num = list_label[counter]
            word_list = sentence.split() 
            for dummy in word_list:
                word.append(dummy)
                word_label.append(num)
                for key, value in lang_dict.items():
                    if value == num:
                        word_language.append(key)
                    else:
                        continue
            counter += 1




        ##############################################################################################
        #data is created as a dictionary. Key = column name and Value = list of data
        #Text has all the word data
        #Language has all of the words
        data = {'Text': word,
                'Language': word_language}


        #creates the desired dataframe from words and its corresponding language
        df = pd.DataFrame(data)


        #Need to feature Extract, this will include turning the text into numerical values
        #One of the Simplest ways to do this
        

        vectorization = CountVectorizer()
        list = []

        #store column values into a list
        #this is converting each word into a numerical value
        for i in df['Text']:
            list.append(i)

        #This is converting words to numerical value
        vectorization.fit(list)
        train = vectorization.transform(list)
        train.toarray()



       #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #Use label encoder instead of original labeling that was done
        label = le.fit_transform(df['Language'])

        #creates train-test splits
        image_train, image_test, label_train, label_test = train_test_split(train, label, test_size=0.3, random_state=42)


        neighbor_list = [1, 3, 5, 10]

        for i in neighbor_list:

            model = KNeighborsClassifier(n_neighbors = i)

            
            model.fit(image_train, label_train)


            #save the model into directory 
            with open('KNN' + str(i) + '_model.pickle', 'wb') as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #save the dictionary with language and numeric values
            with open('KNN' + str(i) + '_labels.pickle', 'wb') as handle:
                pickle.dump(lang_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            #save the test_set in case need to do analysis based off this model later on
            with open('KNN' + str(i) + '_test_set.pickle', 'wb') as handle:
                pickle.dump(image_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

            #save the test_nueric_labels to check against the test_set
            with open('KNN' + str(i) +  '_test_labels.pickle', 'wb') as handle:
                pickle.dump(label_test, handle, protocol=pickle.HIGHEST_PROTOCOL)            

            #save the test_nueric_labels to check against the test_set
            with open('KNN' + str(i) + 'vectorizer.pickle', 'wb') as handle:
                pickle.dump(vectorization, handle, protocol=pickle.HIGHEST_PROTOCOL) 

            with open('KNN' + str(i) + 'labelencoder.pickle', 'wb') as handle:
                pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL) 


#Enter you pathway for this data set
#Dataset is for language, should be a .csv when downloaded
path = '......../Language Detection.csv'
knn = KNN_Language(path)
knn.driver()


