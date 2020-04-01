from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow_core.python.keras.models import load_model

from visualization.graphs import Graphs
from preparer.TextPreparer import PrepareText
from cleaner.TextCleaner import CleanText


class ModelFile(CleanText, PrepareText, Graphs):

    def __init__(self, model, way, way_study, batch_size, epochs, vocab_size = None):
        self.way = way
        self.way_study = way_study
        self.batch_size = batch_size
        self.epochs = epochs
        self.__vocab_size = vocab_size
        self.model = model

    def read(self):
        self.df = pd.read_csv(self.way)
        self.df = self.df.sample(frac=1).reset_index(drop=True)

        self.df_study = pd.read_csv(self.way_study)
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def clean_and_prapare(self):
        self.df = CleanText.clean_category(self.df)
        self.df = CleanText.prepare_text(self.df)

        self.df_study = CleanText.clean_category(self.df_study)
        self.df_study = CleanText.prepare_text(self.df_study)

    def data_to_vect(self):
        tokenizer, textSequences = PrepareText.tokenizer(self.df_study.description)

        y_train = self.df_study.category_code

        self.X_train = self.df.description
        self.y_train = self.df.category_code

        total_unique_words, self.maxSequenceLength = PrepareText.max_count(self.df_study.description, tokenizer)
        self.encoder, self.num_classes = PrepareText.num_classes(y_train)

        return total_unique_words

    @property
    def vocab_size(self):
        return self.__vocab_size

    @vocab_size.setter
    def vocab_size(self,total_unique_words):
        self.__vocab_size = round(total_unique_words / 10)

    def data_final(self):
        tokenizer = Tokenizer(num_words=self.__vocab_size)
        tokenizer.fit_on_texts(self.df_study.description)

        X_train = tokenizer.texts_to_sequences(self.X_train)

        self.X_train = sequence.pad_sequences(X_train, maxlen=self.maxSequenceLength)

        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)

    def train_model(self):
        self.model = load_model(self.model)
        print('Тренируем модель...')
        print(self.X_train, self.y_train)
        self.history = self.model.fit(self.X_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs)

        self.model.save('../model/save/model_new.h5')