import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow_core.python.keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from preparer.TextPreparer import PrepareText
from cleaner.TextCleaner import CleanText
from abc import ABC, abstractmethod


class Model(ABC):

    def __init__(self, model, way_study, batch_size, epochs, vocab_size=None):
        self.model = model
        self.way_study = way_study
        self.batch_size = batch_size
        self.epochs = epochs
        self.__vocab_size = vocab_size

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def clean_and_prapare(self):
        pass

    @abstractmethod
    def data_to_vect(self):
        pass


class ModelFile(Model, CleanText, PrepareText):

    def __init__(self, way, model, way_study, batch_size, epochs, vocab_size):
        Model.__init__(self, model, way_study, batch_size, epochs, vocab_size)
        self.way = way

    def read(self):
        self.df = pd.read_csv(self.way)
        self.df = self.df.sample(frac=1).reset_index(drop=True)

        self.df_study = pd.read_csv(self.way_study)

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
    def vocab_size(self, total_unique_words):
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
        self.history = self.model.fit(self.X_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs)

        self.model.save('../model/save/model_new.h5')

class ModelSingle(Model, CleanText, PrepareText):

    def __init__(self, text, category,  model, way_study, batch_size, epochs, vocab_size):
        Model.__init__(self, model, way_study, batch_size, epochs, vocab_size)
        self.text = text
        self.category = category

    def read(self):
        text_list = list()
        for x in range(6):
            text_list.append(self.text)
        self.df = pd.DataFrame({'text': text_list})
        self.X_train = self.df.text

        self.df_study = pd.read_csv(self.way_study)

    def clean_and_prapare(self):
        self.df = CleanText.prepare_text(self.df)

        self.df_study = CleanText.clean_category(self.df_study)
        self.df_study = CleanText.prepare_text(self.df_study)

    def data_to_vect(self):
        tokenizer, textSequences = PrepareText.tokenizer(self.df_study.description)

        X_train = self.df.description

        y_train = self.df_study.category_code

        total_unique_words, maxSequenceLength = PrepareText.max_count(self.df_study.description, tokenizer)
        self.encoder, self.num_classes = PrepareText.num_classes(y_train)
        self.__vocab_size = round(total_unique_words / 10)
        tokenizer = Tokenizer(num_words=self.__vocab_size)
        tokenizer.fit_on_texts(self.df.description)

        X_train = tokenizer.texts_to_sequences(X_train)
        self.X_train = sequence.pad_sequences(X_train, maxlen=maxSequenceLength)

        if self.category != None:
            cat = '../data/category_with_received_data.csv'
            with open(cat, 'a') as f:
                f.write('"{}",{}\n'.format(self.text, self.category))


    def predict(self):
        self.model = load_model(self.model)
        self.model.predict(self.X_train)

        text_labels = self.encoder.classes_

        prediction = self.model.predict([self.X_train])
        predicted_label = text_labels[np.argmax(prediction)]

        return predicted_label


    def train(self):
        dict_changes = {
            'Positive': np.array([1., 0., 0., 0., 0., 0.]),
            'Negative': np.array([0., 1., 0., 0., 0., 0.]),
            'Hotline': np.array([0., 0., 1., 0., 0., 0.]),
            'Hooligan': np.array([0., 0., 0., 1., 0., 0.]),
            'Offer': np.array([0., 0., 0., 0., 1., 0.]),
            'SiteAndCoins': np.array([0., 0., 0., 0., 0., 1.]),
        }
        y_train = []
        for x in range(6):
            y_train.append(dict_changes[self.category])

        self.model = load_model(self.model)
        self.model.fit(self.X_train, np.array(y_train), batch_size=1,
                  epochs=1)
        self.model.save('../model/save/model_new.h5')
