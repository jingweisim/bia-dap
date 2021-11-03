#read csv file
import pandas as pd
df = pd.read_csv('With Text Augmentation.csv')

df = df[["Text", "Label"]]

#drop rows with empty cells
import numpy as np
for column in df.columns:
    df[column].replace('', np.nan, inplace=True)
df = df.dropna(axis=0)
df = df.reset_index(drop=True)

from sklearn.model_selection import train_test_split
X = df['Text']
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,stratify=y,random_state=19)

from sklearn.feature_extraction.text import TfidfVectorizer

#https://www.youtube.com/watch?v=k5jk1RrCp-o
#create the TF-IDF object
tfidf = TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))

#transform the train dataset
tfidf_train = tfidf.fit_transform(X_train)

#transform the test dataset
tfidf_test = tfidf.transform(X_test)

print('TF-IDF train:',tfidf_train.shape)
print('TF-IDF test:',tfidf_test.shape)

from sklearn.naive_bayes import MultinomialNB

#create model object
mnb = MultinomialNB()

#fit the model to the TF-IDF features
mnb_tfidf = mnb.fit(tfidf_train,y_train)

#predicting the model for TF-IDF features
mnb_tfidf_predict = mnb.predict(tfidf_test)

#check the accuracy score for tfidf features
from sklearn.metrics import accuracy_score
mnb_tfidf_score = accuracy_score(y_test,mnb_tfidf_predict)
print("Naive bayes TF-IDF accuracy score:",mnb_tfidf_score)

from sklearn.metrics import classification_report
mnb_tfidf_report = classification_report(y_test,mnb_tfidf_predict,target_names=['0','1'])
print('TF-IDF accuracy score:')
print(mnb_tfidf_report)
