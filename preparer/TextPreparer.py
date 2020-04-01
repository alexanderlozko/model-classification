import numpy as np
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder


class PrepareText:
    """
    Prepare text for model
    """

    @staticmethod
    def tokenizer(descriptions):
        """
        Vectorize a text corpus, by turning each text into either a sequence of integers

        :param descriptions: Clean text
        :return: tokenizer - text tokenization utility class, textSequences - Converts a text to a sequence of words (or tokens)
        """

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(descriptions.tolist())
        textSequences = tokenizer.texts_to_sequences(descriptions.tolist())

        return tokenizer, textSequences

    @staticmethod
    def max_count(descriptions, tokenizer):
        """
        Count words in the text

        :param descriptions: Clean text
        :param tokenizer: Text tokenization utility class
        :return: total_unique_words - total unique words, maxSequenceLength - max amount of words in the longest text
        """

        max_words = 0
        for desc in descriptions.tolist():
            words = len(desc.split())
            if words > max_words:
                max_words = words
        print('Максимальное количество слов в самом длинном описании заявки: {} слов'.format(max_words))

        total_unique_words = len(tokenizer.word_counts)
        print('Всего уникальных слов в словаре: {}'.format(total_unique_words))

        maxSequenceLength = max_words

        return total_unique_words, maxSequenceLength

    @staticmethod
    def num_classes(y_train):
        """
        Count categories

        :param y_train: Training labels
        :param y_test: Test labels
        :return: encoder - encode target labels, num_classes - amount of classes
        """

        encoder = LabelEncoder()
        encoder.fit(y_train)
        y_train = encoder.transform(y_train)
        #y_test = encoder.transform(y_test)

        num_classes = np.max(y_train) + 1

        print('Количество категорий для классификации: {}'.format(num_classes))

        return encoder, num_classes

    @staticmethod
    def load_data_from_arrays(descriptions, category, train_test_split=0.8):
        """
        Divide data into training and test sets

        :param descriptions: Clean text
        :param category: Categories
        :param train_test_split: Training set size
        :return: Data divided into training and test
        """

        data_size = len(descriptions)
        test_size = int(data_size - round(data_size * train_test_split))
        print("Test size: {}".format(test_size))

        print("\nTraining set:")
        x_train = descriptions[test_size:]
        print("\t - x_train: {}".format(len(x_train)))
        y_train = category[test_size:]
        print("\t - y_train: {}".format(len(y_train)))

        print("\nTesting set:")
        x_test = descriptions[:test_size]
        print("\t - x_test: {}".format(len(x_test)))
        y_test = category[:test_size]
        print("\t - y_test: {}".format(len(y_test)))

        return x_train, y_train, x_test, y_test

    @staticmethod
    def transform_sets(vocab_size, descriptions, X_train, X_test, y_train, y_test, maxSequenceLength, num_classes):
        """
        Transform the data for training and testing in the format we need

        :param vocab_size: 10% of total unique words
        :param descriptions: Clean text
        :param X_train: Training data set
        :param X_test: Test data set
        :param y_train: Training labels
        :param y_test: Test labels
        :param maxSequenceLength: Max amount of words in the longest text
        :param num_classes: Number of classes
        :return: transformed data
        """

        print('Преобразуем описания заявок в векторы чисел...')
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(descriptions)

        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)

        X_train = sequence.pad_sequences(X_train, maxlen=maxSequenceLength)
        X_test = sequence.pad_sequences(X_test, maxlen=maxSequenceLength)

        print('Размерность X_train:', X_train.shape)
        print('Размерность X_test:', X_test.shape)

        print(u'Преобразуем категории в матрицу двоичных чисел '
              u'(для использования categorical_crossentropy)')
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        print('y_train shape:', y_train.shape)
        print('y_test shape:', y_test.shape)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def parameters(df, train_test_split=0.8):
        """
        Combines several function

        :param df: DataFrame
        :param train_test_split: Number to split training data
        :return: Function values
        """

        categories = df.category_code
        descriptions = df.description

        tokenizer, textSequences = PrepareText.tokenizer(descriptions)
        X_train, y_train, X_test, y_test = PrepareText.load_data_from_arrays(descriptions, categories, train_test_split)

        total_unique_words, maxSequenceLength = PrepareText.max_count(descriptions, tokenizer)
        vocab_size = round(total_unique_words / 10)

        encoder, num_classes = PrepareText.num_classes(y_train, y_test)

        return maxSequenceLength, vocab_size, encoder, num_classes

    @staticmethod
    def change_cat(category):
        """
        Turns names of categories into numbers

        :param category: Names of category
        :return: Number of category
        """

        if category == 'Positive':
            return 0
        elif category == 'Negative':
            return 1
        elif category == 'Hotline':
            return 2
        elif category == 'Hooligan':
            return 3
        elif category == 'Offer':
            return 4
        elif category == 'SiteAndCoins':
            return 5