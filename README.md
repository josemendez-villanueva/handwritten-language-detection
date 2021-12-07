# handwritten-language-detection

Image Dataset = NIST Special Database 19 (Download by class) https://www.nist.gov/srd/nist-special-database-19
Language Detection Dataset = https://www.kaggle.com/basilb2s/language-detection


The pathways that are generated need the path before the repository name, also done in Windows.
Need to manually change all the paths for all the files

For the folders that contain the files for the image classification models that need to be trained, those files need an entire pathway change based on 
your data pathway, the language training model files contains the very last extension or name assuming the data that is mentioned is used.


analysis file contains the code if you want to run analysis on your trained models

The file Language Detection the the main code that does the prediction but the models need to be trained before this.

There are the image classification models and the language detection models that need to be trained. The example ones used in the Language Detection.py
is using specifically CNNadamaxE6 for the image classification. Run the CNN.py file to obtain this model and everything else you need. Go into the KNN_Language (Word) 
and run the KNN.py file in order to get the language detection model and everything else you need.

Now you are able to run the Language Detection.py successfully




