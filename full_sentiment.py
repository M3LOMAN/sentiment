import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import re

from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import nltk
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
#Create lemmatizer and stopwords list
russian_stopwords = stopwords.words("russian")
import dataclasses

df = pd.read_csv(r"/C:/Users/Владимир/OneDrive/Документы/sent_analysis/labeled.csv", sep=",")
df['toxic'] = df['toxic'].apply(int)
df_bad = df[df["toxic"] == 1]["comment"]
df_good=df[df["toxic"]==0]["comment"]
sorted_df = df.sort_values(by='toxic')
count_bad =len(df_bad)
count_good=len(df_good)
del sorted_df['toxic']

texts = sorted_df['comment'].values.tolist()
maxWordsCount = 50000

from nltk.tokenize import word_tokenize
nltk.download('punkt')
text_tokens=[]
for i in range(len(texts)):
  text_tokens.append(word_tokenize(texts[i]))
tokens_without_sw =[]
for i in range(len(text_tokens)):
  tokens_without_sw.append([word for word in text_tokens[i] if not word in russian_stopwords])

print(tokens_without_sw[0])
good_texts=[]
for i in range(len(tokens_without_sw)):
  sentence=""
  for word in tokens_without_sw[i]:
    sentence+= str(word) + " "
  good_texts.append(sentence)

tokenizer = Tokenizer(num_words=maxWordsCount, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»', split=' ', lower=True, char_level=False)
tokenizer.fit_on_texts(good_texts)
dist = list(tokenizer.word_counts.items())

max_text_len = 10
data = tokenizer.texts_to_sequences(good_texts)
data_pad = pad_sequences(data, maxlen=max_text_len)
print(data_pad)

X = data_pad
Y = np.array([[1, 0]]*count_good + [[0, 1]]*count_bad)
print(X.shape, Y.shape)

indeces = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
X = X[indeces]
Y = Y[indeces]

model = Sequential()
model.add(Embedding(maxWordsCount, 128, input_length = max_text_len))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(0.0001))
history = model.fit(X, Y, batch_size=32, epochs=15)
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

def sequence_to_text(list_of_indices):
    words = [reverse_word_map.get(letter) for letter in list_of_indices]
    return(words)

t = " мразь училка".lower()
data = tokenizer.texts_to_sequences([t])
data_pad = pad_sequences(data, maxlen=max_text_len)
print( sequence_to_text(data[0]) )

res = model.predict(data_pad)
print(res, np.argmax(res), sep='\n')