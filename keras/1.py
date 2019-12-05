# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 20:36:36 2019

@author: Inkling
"""
from keras.models import load_model
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from keras import preprocessing
import os


# Load train data
df1=pd.read_csv('train2/sigchi_training.csv',encoding = "ISO-8859-1")
df2=pd.read_csv('train2/sigcse_training.csv',encoding = "ISO-8859-1")
df3=pd.read_csv('train2/cikm_training.csv',encoding = "ISO-8859-1")
df4=pd.read_csv('train2/sigkdd_training.csv',encoding = "ISO-8859-1")
df5=pd.read_csv('train2/siggraph_training.csv',encoding = "ISO-8859-1")
df6=pd.read_csv('train2/sigir_training.csv',encoding = "ISO-8859-1")
df7=pd.read_csv('train2/www_training.csv',encoding = "ISO-8859-1")

# Load validation data
df11=pd.read_csv('validation/chi_test.csv',encoding = "ISO-8859-1")
df12=pd.read_csv('validation/cse_test.csv',encoding = "ISO-8859-1")
df13=pd.read_csv('validation/cikm_test.csv',encoding = "ISO-8859-1")
df14=pd.read_csv('validation/kdd_test.csv',encoding = "ISO-8859-1")
df15=pd.read_csv('validation/siggraph_test.csv',encoding = "ISO-8859-1")
df16=pd.read_csv('validation/sigir_test.csv',encoding = "ISO-8859-1")
df17=pd.read_csv('validation/www_test.csv',encoding = "ISO-8859-1")

# Set labels
df1['label']='0'
df2['label']='1'
df3['label']='2'
df4['label']='3'
df5['label']='4'
df6['label']='5'
df7['label']='6'
df11['label']='0'
df12['label']='1'
df13['label']='2'
df14['label']='3'
df15['label']='4'
df16['label']='5'
df17['label']='6'

df1=df1[['Column2','label']]
df2=df2[['Column2','label']]
df3=df3[['Column2','label']]
df4=df4[['Column2','label']]
df5=df5[['Column2','label']]
df6=df6[['Column2','label']]
df7=df7[['Column2','label']]
df11=df11[['Column2','label']]
df12=df12[['Column2','label']]
df13=df13[['Column2','label']]
df14=df14[['Column2','label']]
df15=df15[['Column2','label']]
df16=df16[['Column2','label']]
df17=df17[['Column2','label']]

# Load test data
sample=pd.read_csv('sample.csv',encoding = "ISO-8859-1")
sample=sample['Column2']


# Merge train data
df=pd.concat([df1,df2,df3,df4,df5,df6,df7,df11,df12,df13,df14,df15,df16,df17])

# Remove 'NA'
df=df.dropna(axis='rows')

# Other option
pd.set_option('display.max_rows', None)

# Set Keras
from keras.preprocessing.text import Tokenizer
vocabulary = 20000
tokenizer = Tokenizer(num_words=vocabulary) 
tokenizer.fit_on_texts(df['Column2'].tolist())
word_index = tokenizer.word_index
sequences_train = tokenizer.texts_to_sequences(df['Column2'].tolist())

# Tokenize sample data
sequences_sample=tokenizer.texts_to_sequences(sample.tolist())

# Maintaince length
z=preprocessing.sequence.pad_sequences(sequences_sample, maxlen=200)


from keras import preprocessing
from keras.utils.np_utils import to_categorical

x=preprocessing.sequence.pad_sequences(sequences_train, maxlen=200)
y=to_categorical(df.label)

# Split train and test data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state =0)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

# Set up keras model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
embedding_dim = 100
model = Sequential()
model.add(Embedding(20000, embedding_dim, input_length=200)) 
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2,return_sequences=False)))
model.add(Dense(7, activation='softmax'))
model.summary()

# Load trained model
#model.load_weights('md.h5')

from keras import optimizers

es = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True)
rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=3, min_lr=1e-6)
callback_list = [es, rlrop]
model.compile(optimizer=optimizers.RMSprop(lr=1E-3), loss='categorical_crossentropy', metrics=['acc'])

# Train model
history=model.fit(x_train,y_train, epochs=5,batch_size=32, validation_data=(x_test,y_test), callbacks=callback_list)

############
############
#model = load_model('md.h5')

# Save the trained model
#model.save('md_title.h5')

# Plot train-validation result
import matplotlib.pylab as plt

acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Predict new data
for i in model.predict(z):
    print(np.argmax(i))