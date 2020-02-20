import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
  
#cleaning the texts  
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review=review.lower()
    #remove significatent word to predict is it positive or negative review
    #list created
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #steaming
    #joining back
    review= ' '.join(review)
    corpus.append(review)
    
#creating the bag of list model
from sklearn.feature_extraction.text import CountVectorizer    
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray() 
y=dataset.iloc[:,1].values   
#training(spersitity)

#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)








