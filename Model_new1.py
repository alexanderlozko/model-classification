from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from visualization.graphs import Graphs
from preparer.TextPreparer import PrepareText
from cleaner.TextCleaner import CleanText


class Model(CleanText, PrepareText, Graphs):

    def __init__(self, way, way_study, batch_size, epochs, vocab_size = None):
        self.way = way
        self.way_study = way_study
        self.batch_size = batch_size
        self.epochs = epochs
        self.__vocab_size = vocab_size

    def read(self):
        self.df = pd.read_csv(self.way)
        self.df = self.df.sample(frac=1).reset_index(drop=True)

        self.df_study = pd.read_csv(self.way_study)
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def clean_and_prapare(self):
        self.df = CleanText.clean_category(self.df)
        self.df = CleanText.prepare_text(self.df)



    def data_to_vect(self):
        tokenizer, textSequences = PrepareText.tokenizer(self.df.descriptions)
        #self.X_train, self.y_train, self.X_test, self.y_test = PrepareText.load_data_from_arrays(self.df.descriptions, self.df.category_code, train_test_split=0.8)

        self.X_train = self.df.descriptions
        self.y_train = self.df.category_code

        self.total_unique_words, self.maxSequenceLength = PrepareText.max_count(self.df.descriptions, tokenizer)
        self.encoder, self.num_classes = PrepareText.num_classes(self.y_train, self.y_test)

    @property
    def vocab(self):
        return self.__vocab_size

    @vocab.setter
    def vocab(self, total_unique_words):
        self.__vocab_size = round(total_unique_words / 10)

    def data_final(self):
        self.X_train, self.X_test, self.y_train, self.y_test = PrepareText.transform_sets(self.__vocab_size,
                                                                                          self.df.descriptions, self.X_train,
                                                                                          self.X_test, self.y_train, self.y_test,
                                                                                          self.maxSequenceLength,
                                                                                          self.num_classes)

    def create_model(self):
        print('Собираем модель...')
        self.model = Sequential()
        self.model.add(Embedding(self.vocab_size, self.maxSequenceLength))
        self.model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(self.num_classes, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print(self.model.summary())

    def train_model(self):
        print('Тренируем модель...')
        self.history = self.model.fit(self.X_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  validation_data=(self.X_test, self.y_test))

        score = self.model.evaluate(self.X_test, self.y_test,
                               batch_size=self.batch_size, verbose=1)

        print()
        print(u'Оценка теста: {}'.format(score[0]))
        print(u'Оценка точности модели: {}'.format(score[1]))
        self.model.save('.../model/save/model_new.h5')

    def predict(self):
        Graphs.plot_efficiency(self.epochs, self.history)

        text_labels = self.encoder.classes_

        for i in range(len(self.y_test)):
            prediction = self.model.predict(np.array([self.X_test[i]]))
            predicted_label = text_labels[np.argmax(prediction)]
            print('====================================')
            print('Правильная категория: {}'.format(self.y_test[i]))
            print("Определенная моделью категория: {}".format(predicted_label))


        y_softmax = self.model.predict(self.X_test)

        y_test_1d = []
        y_pred_1d = []

        for i in range(len(self.y_test)):
            probs = self.y_test[i]
            index_arr = np.nonzero(probs)
            one_hot_index = index_arr[0].item(0)
            y_test_1d.append(one_hot_index)

        for i in range(0, len(y_softmax)):
            probs = y_softmax[i]
            predicted_index = np.argmax(probs)
            y_pred_1d.append(predicted_index)

        text_labels = self.encoder.classes_
        cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
        plt.figure(figsize=(48,40))
        Graphs.plot_confusion_matrix(cnf_matrix, classes=text_labels, title="Confusion matrix")

