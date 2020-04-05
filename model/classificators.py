import pandas as pd
from abc import ABC, abstractmethod
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from cleaner.TextCleaner import CleanText
from preparer.TextPreparer import PrepareText


class Model(ABC):

    def __init__(self,  way_study):
        self.way_study = way_study

    def template_method(self):
        self.read()
        self.clean_and_prapare()
        self.data_to_vect()
        self.RandomForestRegressor()
        self.SVСModel()
        self.LogisticRegressionModel()

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def clean_and_prapare(self):
        pass

    @abstractmethod
    def data_to_vect(self):
        pass

    def RandomForestRegressor(self):
        pass

    def SVСModel(self):
        pass

    def LogisticRegressionModel(self):
        pass

class ModelFile(Model, CleanText, PrepareText):

    def __init__(self, way_study):
        Model.__init__(self,  way_study)

    def read(self):
        self.df_study = pd.read_csv(self.way_study)

    def clean_and_prapare(self):
        self.df_study = CleanText.clean_category(self.df_study)
        self.df_study = CleanText.prepare_text(self.df_study)

    def data_to_vect(self):
        tokenizer, textSequences = PrepareText.tokenizer(self.df_study.description)

        X_train, y_train, X_test, y_test = PrepareText.load_data_from_arrays(self.df_study.description, self.df_study.category_code,
                                                                                train_test_split=0.8)

        total_unique_words, maxSequenceLength = PrepareText.max_count(self.df_study.description, tokenizer)
        vocab_size = round(total_unique_words / 10)

        self.Y_train = y_train
        self.Y_test = y_test
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(self.df_study.description)

        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)

        self.X_train = sequence.pad_sequences(X_train, maxlen=maxSequenceLength)
        self.X_test = sequence.pad_sequences(X_test, maxlen=maxSequenceLength)

class RandomForestRegressor(ModelFile):

    def RandomForestRegressor(self):
        classifier = RandomForestClassifier(criterion='gini', max_depth=21, n_estimators=1)

        classifier.fit(self.X_train, self.Y_train)
        y_pred = classifier.predict(self.X_test)

        report = classification_report(self.Y_test, y_pred)
        print(report)

        matrix = confusion_matrix(self.Y_test, y_pred)
        print(matrix)

class SVСModel(ModelFile):

    def SVСModel(self):
        classifier = SVC()

        classifier.fit(self.X_train, self.Y_train)
        y_pred = classifier.predict(self.X_test)

        report = classification_report(self.Y_test, y_pred)
        print(report)

        matrix = confusion_matrix(self.Y_test, y_pred)
        print(matrix)

class LogisticRegressionModel(ModelFile):

    def LogisticRegressionModel(self):
        classifier = LogisticRegression(multi_class='ovr', penalty='none', solver='sag')

        classifier.fit(self.X_train, self.Y_train)
        y_pred = classifier.predict(self.X_test)

        report = classification_report(self.Y_test, y_pred)
        print(report)

        matrix = confusion_matrix(self.Y_test, y_pred)
        print(matrix)

def run(abstract_class:Model):
    abstract_class.template_method()

if __name__ == "__main__":
    print("RandomForestRegressor:")
    run(RandomForestRegressor('../data/category_with_received_data.csv'))
    print("")

    print("SVM:")
    run(SVСModel('../data/category_with_received_data.csv'))
    print("")

    print("LogisticRegression:")
    run(LogisticRegressionModel('../data/category_with_received_data.csv'))
