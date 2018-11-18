import os
os.environ['KERAS_BACKEND']='theano'
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import os
os.environ['KERAS_BACKEND']='theano'
import numpy as np

import re
import pandas as pd
from bs4 import BeautifulSoup

from sklearn.ensemble import VotingClassifier

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.25


def clean_str(string):
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

data_train = pd.read_csv('label_question.csv', sep='\t\t\t\t\t')
print data_train.shape
texts = []
labels = []

for idx in range(data_train.body.shape[0]):
    if data_train.body[idx]==" ":
        continue
    text = BeautifulSoup(data_train.body[idx], "lxml")
    texts.append(clean_str(text.get_text().encode('ascii','ignore')))
    labels.append(data_train.label[idx])


print labels

embeddings_index = {}
f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

tokenizer = Tokenizer(nb_words=None)
sequences = tokenizer.fit_on_texts(texts)
documents = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
token_count = len(word_index)+1
data = pad_sequences(documents, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

# labels = to_categorical(np.asarray(labels))

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = np.array(labels)[indices]
# SPLIT DATA , ONE PART FOR TRAIN , THE OTHER PART FOR PREDICT
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_test = data[-nb_validation_samples:]
y_test = labels[-nb_validation_samples:]
clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial',
                      random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()
eclf1 = VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

eclf1 = eclf1.fit(x_train, y_train)

y_pred = eclf1.predict(x_test)


score = accuracy_score(y_test, y_pred)

print score



