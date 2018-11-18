import os
os.environ['KERAS_BACKEND']='theano'
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dropout, Dense
from keras.layers import Embedding, GlobalMaxPooling1D
from keras.utils.np_utils import to_categorical
from keras import regularizers
import re
import pandas as pd
from bs4 import BeautifulSoup
from keras.layers import Dense, Input
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input
from keras.layers import  Embedding, LSTM, Bidirectional
from keras.models import  Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from Attention import Attention_layer

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


print x_train.shape
print x_test.shape

embedding_matrix = np.zeros((token_count, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
      embedding_matrix[i,:] = embedding_vector

embedding_layer = Embedding(
  token_count,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=MAX_SEQUENCE_LENGTH,
  trainable=False)


def create_cnn_model():
    model = Sequential()
    model.add(embedding_layer)#, input_shape= (token_count, EMBEDDING_DIM))
    model.add(Conv1D(filters = 64, kernel_size = 4, padding = 'same', activation='relu'))
    #input_shape=(token_count,EMBEDDING_DIM)))
    model.add(MaxPooling1D())#kernel_size=500))
    model.add(Conv1D(filters = 128, kernel_size = 3, padding = 'same',  activation='relu',
                     kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D())
    model.add(Conv1D(filters = 256, kernel_size = 2, padding = 'same', activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    #model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    # model.fit(x_train, y_train, validation_data=(x_val,y_val),batch_size=16, epochs=30, validation_split = VALIDATION_SPLIT)
    return model


def create_bilstm_attention_model():
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    l_gru = Bidirectional(LSTM(100, return_sequences=True))(embedded_sequences)
    l_att = Attention_layer()(l_gru)
    dense_1 = Dense(100, activation='tanh')(l_att)
    dense_2 = Dense(1, activation='softmax')(dense_1)
    model = Model(sequence_input, dense_2)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    # model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=40, batch_size=16)
    model.summary()
    return model


clf_cnn = KerasClassifier(build_fn=create_cnn_model,batch_size=2000, nb_epoch=1)
clf_rnn = KerasClassifier(build_fn=create_bilstm_attention_model,nb_epoch=1, batch_size=2000)


eclf1 = VotingClassifier(estimators=[ ('clf_cnn1', clf_cnn),
                                      ('clf_cnn2', clf_cnn)
                                    ])

eclf1.fit(x_train, y_train)


y_pred = eclf1.predict(x_test)

score = accuracy_score(y_test, y_pred)

print(score)
