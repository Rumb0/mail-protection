#!/usr/bin/env python
# coding: utf-8

# In[1]:
import re

from pathlib import PurePath

import pandas as pd
import numpy as np

from gensim.models.phrases import Phraser, Phrases, ENGLISH_CONNECTOR_WORDS

from gensim.parsing.preprocessing import (
    preprocess_string,
    strip_tags,
    strip_punctuation,
    strip_multiple_whitespaces,
    remove_stopwords,
    strip_short
)

import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer
)

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    roc_curve,
    RocCurveDisplay,
    accuracy_score,
    recall_score,
    precision_score
)

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing import sequence, text

import explore_data
# import vectorize_data


# In[2]:
matplotlib.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')
tqdm.pandas()


# In[53]:
data = pd.read_csv('data.csv').set_index('id')
data


# In[ ]:
data.shape


# In[ ]:
deduplicated_data = data.drop_duplicates(subset=['text', 'is_spam'])
deduplicated_data.shape


# In[ ]:
unique_text_data = data.drop_duplicates(subset=['text'])
unique_text_data.shape


# In[ ]:
dub2 = deduplicated_data[~deduplicated_data.index.isin(unique_text_data.index)]
dub2


# In[ ]:
neg, _ = deduplicated_data[deduplicated_data.is_spam == False].shape
pos, _ = deduplicated_data[deduplicated_data.is_spam == True].shape
neg, pos


# In[ ]:
double_verdict_texts = deduplicated_data[deduplicated_data.text.isin(dub2.text)].sort_values(by=['text'])
double_verdict_texts


# In[ ]:
double_verdict_texts_todelete = double_verdict_texts[double_verdict_texts.is_spam == False]
double_verdict_texts_todelete


# In[ ]:
deduplicated_data = data[~data.index.isin(double_verdict_texts_todelete.index)].drop_duplicates(subset=['text'])
deduplicated_data


# In[ ]:
neg, _ = deduplicated_data[deduplicated_data.is_spam == False].shape
pos, _ = deduplicated_data[deduplicated_data.is_spam == True].shape
neg, pos


# In[ ]:
data = deduplicated_data[['text', 'is_spam']]
data


# In[18]:
DEFAULT_FILTERS = [
    lambda x: x.lower(),
    strip_tags,
    strip_punctuation,
    strip_multiple_whitespaces,
    remove_stopwords,
    strip_short
]

def make_bigrams(texts, bigram):
    return [bigram[doc] for doc in texts]

def make_trigrams(texts, bigram, trigram):
    return [trigram[bigram[doc]] for doc in texts]

def preproc_doc(docs):
    texts = [[text for text in preprocess_string(doc, filters=DEFAULT_FILTERS) if 3 < len(text) < 100 and not re.search(r'\d', text)] for doc in docs]
    bigrams_phrases = Phrases(texts, min_count=20, threshold=200)
    # trigrams_phrases = Phrases(bigrams_phrases[texts], threshold=100)
    bigram = Phraser(bigrams_phrases)
    # trigram = Phraser(trigrams_phrases)
    data_bigrams = make_bigrams(texts, bigram)
    # data_bigrams_trigrams = make_trigrams(data_bigrams, bigram, trigram)
    return [' '.join(text) for text in data_bigrams]


# In[ ]:
np.array(preprocess_string(data.text.astype(str)[1734141], filters=DEFAULT_FILTERS))


# In[ ]:
bd_searcher = re.compile(r'\d')


# In[ ]:
data["tokens"] = data.text.astype(str).progress_apply(lambda text: ' '.join(preprocess_string(text, filters=DEFAULT_FILTERS)))
data


# In[ ]:
data = data[['text', 'is_spam', 'tokens']].dropna(how='any')
data


# In[19]:
from polyglot.text import Text, Word
from polyglot.detect.base import logger as polyglot_logger

# we filter out low confidence detections
# so we don't need thouse warnings
polyglot_logger.setLevel("ERROR")

# deterministic, slow algorythm
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# spacy needs larger corpus than we have
# so we don't use it

def detect_lang(text, conf_threshold = 80):
    text = str(text)

    try:
        l = Text(text).language
    except:
        l = None

        try:
            code = detect(text)
        except:
            return(pd.NA)

    if l is not None:
        if l.confidence < conf_threshold:
            try:
                code = detect(text)
            except:
                return(pd.NA)
        else:
            code = l.code

    return(code)


# In[ ]:
## this takes 25-30 mins to complete

# data["lang code"] = data.text.progress_apply(detect_lang)
# data.to_csv(PurePath('data_processed.csv'))


# In[3]:
data = pd.read_csv('data_processed.csv').set_index('id')
data = data.dropna(how='any')
data


# In[4]:
t = data.groupby('lang code').count().text.nlargest(11)
t = t[t.index != 'ru']

fig, ax = plt.subplots()
ax.bar(np.array(t.index), np.array(t))

plt.title('Распределение языков в наборе данных')
plt.ylabel('Число писем')
plt.xlabel('Код языка')

plt.show()


# In[5]:
data_en = data[data['lang code'] == 'en']
data_en


# In[6]:
labels = 'Спам', 'Не спам'
sizes = [data_en[data_en.is_spam == True].shape[0], data_en[data_en.is_spam == False].shape[0]]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%')
ax1.axis('equal')

plt.show()


# In[7]:
NGRAM_RANGE = (1, 2)  # Use 1-grams + 2-grams.
TOP_K = 20000
TOKEN_MODE = 'word'
MIN_DOCUMENT_FREQUENCY = 2
MAX_SEQUENCE_LENGTH = 500


def ngram_vectorize(train_texts, train_labels, val_texts):
    kwargs = {
            'ngram_range': NGRAM_RANGE,
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': TOKEN_MODE,
            'min_df': MIN_DOCUMENT_FREQUENCY,
    }
    vectorizer = TfidfVectorizer(**kwargs)

    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts)

    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train)
    x_val = selector.transform(x_val)

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    return (vectorizer,selector),  x_train.toarray(), x_val.toarray()


# In[8]:
train_data, test_data = train_test_split(
    data_en,
    train_size = 0.8,
    test_size = 0.2,
    random_state = 42
)

train_texts = np.array(train_data.tokens)
train_labels = tf.cast(train_data.is_spam.apply(lambda x: 1 if x else 0), tf.int8)
val_texts = np.array(test_data.tokens)
val_labels = tf.cast(test_data.is_spam.apply(lambda x: 1 if x else 0), tf.int8)

vectorizer, x_train, x_val = ngram_vectorize(train_texts, train_labels, val_texts)


# In[9]:
def mlp_model(layers, units, dropout_rate, input_shape):

    op_units = 1
    op_activation = 'sigmoid'

    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    for _ in range(layers-1):
        model.add(Dense(units = units, activation='relu'))
        model.add(Dropout(rate = dropout_rate))

    model.add(Dense(units = op_units, activation = op_activation))
    return model


# In[10]:
def train_ngram_model(train_texts,
                      train_labels,
                      val_texts,
                      val_labels,
                      x_train,
                      x_val,
                      learning_rate = 1e-3,
                      epochs = 100,
                      batch_size = 128,
                      layers = 2,
                      units = 64,
                      dropout_rate = 0.2,
                      num_classes = 2):

    model = mlp_model(layers=layers,
                      units=units,
                      dropout_rate=dropout_rate,
                      input_shape=x_train.shape[1:])

    loss = 'binary_crossentropy'
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    ]

    history = model.fit(
        x_train,
        train_labels,
        epochs = epochs,
        callbacks = callbacks,
        validation_data = (x_val, val_labels),
        verbose = 2,
        batch_size = batch_size
    )

    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    return model, history


# In[11]:
model_name = 'MLP_L2_128_sg.h5'


# In[12]:
# this requires 12GB ram
model, history = train_ngram_model(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    x_train,
    x_val,
    batch_size = 512,
    learning_rate = 1e-3,
    layers = 2,
    units = 128,
    dropout_rate = 0.2
)
model.save(model_name)


# In[46]:
model = tf.keras.models.load_model(model_name)


# In[13]:
model.summary()


# In[14]:
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Функция потерь при обучении')
plt.ylabel('Функция потерь')
plt.xlabel('Эпоха (итерация обучения)')
plt.legend(['Тренировочная выборка', 'Проверочная выборка'], loc='upper left')
plt.show()


# In[15]:
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('Точность модели при обучении')
plt.ylabel('Точность')
plt.xlabel('Эпоха (итерация обучения)')
plt.legend(['Тренировочная выборка', 'Проверочная выборка'], loc='upper left')
plt.show()


# In[26]:
y_test = tf.cast(test_data.is_spam.apply(lambda x: 1 if x else 0), tf.int8)
y_test


# In[27]:
y_pred_raw = model.predict(x_val)
y_pred_raw


# In[28]:
(
    loss,
    accuracy
) = model.evaluate(x_val, y_test)


# In[29]:
print(f"{accuracy = }")
# print(f"{precision = }")
# print(f"{recall = }")


# In[30]:
y_test = np.array(y_test, dtype=float)
y_pred_raw = np.array(y_pred_raw)


# In[31]:
fpr, tpr, thresholds = roc_curve(y_test, y_pred_raw, pos_label = 1.)


# In[32]:
RocCurveDisplay.from_predictions(y_test, y_pred_raw)
plt.show()


# In[33]:
PrecisionRecallDisplay.from_predictions(y_test, y_pred_raw)
plt.show()


# In[34]:
accuracy = []
recall = []
precision = []

for thres in thresholds:
    tmp_pred = np.where(y_pred_raw > thres, 1, 0)
    accuracy.append(accuracy_score(y_test, tmp_pred))
    recall.append(recall_score(y_test, tmp_pred))
    precision.append(precision_score(y_test, tmp_pred))

target_data = pd.concat([pd.Series(thresholds), pd.Series(fpr), pd.Series(tpr), pd.Series(accuracy), pd.Series(recall), pd.Series(precision)],
                        axis = 1)

target_data.columns = ['Threshold', 'FPR', 'TPR', 'Accuracy', 'Precision', 'Recall']
target_data.sort_values(by ='Accuracy', ascending = False, inplace = True)
target_data.reset_index(drop = True,inplace = True)


# In[35]:
target_data


# In[36]:
target_threshold = target_data[target_data.Accuracy == target_data.Accuracy.max()].Threshold[0]
target_threshold


# In[37]:
target_data[target_data.Accuracy == target_data.Accuracy.max()]


# In[38]:
y_test = np.array(y_test, dtype=int)
y_pred = np.where(y_pred_raw > target_threshold, 1, 0)


# In[39]:
disp = ConfusionMatrixDisplay(
    confusion_matrix = np.array(tf.math.confusion_matrix(y_test, y_pred))
)
disp.plot()
plt.show()


# In[20]:
enron_data = pd.read_csv('enron6/enron.csv')
enron_data = enron_data.drop_duplicates(subset=['text'])
enron_data


# In[21]:
enron_data["tokens"] = enron_data.text.astype(str).progress_apply(lambda text: ' '.join(preprocess_string(text, filters=DEFAULT_FILTERS)))
enron_data = enron_data[['text', 'is_spam', 'tokens']].dropna(how='any')
enron_data = enron_data.drop_duplicates(subset=['tokens'])
enron_data


# In[22]:
enron_data["lang code"] = enron_data.text.progress_apply(detect_lang)
enron_data_en = enron_data[enron_data["lang code"] == 'en']
enron_data_en.to_csv(PurePath('enron_processed.csv'))
enron_data_en


# In[23]:
enron_x_val = np.array(enron_data_en.tokens)
enron_y_val = tf.cast(enron_data_en.is_spam.apply(lambda x: 1 if x else 0), tf.int8)

enron_x_val = vectorizer[0].transform(enron_x_val)
enron_x_val = vectorizer[1].transform(enron_x_val).toarray()


# In[24]:
(
    loss,
    accuracy
) = model.evaluate(enron_x_val, enron_y_val)

accuracy


# In[40]:
enron_y_pred = model.predict(enron_x_val)
enron_y_pred = np.where(enron_y_pred > target_threshold, 1, 0)


# In[41]:
disp = ConfusionMatrixDisplay(
    confusion_matrix = np.array(tf.math.confusion_matrix(enron_y_val, enron_y_pred))
)

disp.plot()
plt.show()
