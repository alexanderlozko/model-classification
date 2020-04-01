from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import way
from visualization.graphs import Graphs
from preparer.TextPreparer import PrepareText


class ModelFile(PrepareText, Graphs):

    df = pd.read_pickle(way.pickle1)
    df = df.sample(frac=1).reset_index(drop=True)

    categories = df.category_code
    descriptions = df.description

    tokenizer, textSequences = PrepareText.tokenizer(descriptions)
    X_train, y_train, X_test, y_test = PrepareText.load_data_from_arrays(descriptions, categories, train_test_split=0.8)

    total_unique_words, maxSequenceLength = PrepareText.max_count(descriptions, tokenizer)
    vocab_size = round(total_unique_words/10)

    encoder, num_classes = PrepareText.num_classes(y_train, y_test)

    X_train, X_test, y_train, y_test = PrepareText.transform_sets(vocab_size, descriptions ,X_train, X_test, y_train, y_test, maxSequenceLength, num_classes)
    print(X_train[0], y_train[0], X_train, y_train)
    # максимальное количество слов для анализа
    max_features = vocab_size

    print('Собираем модель...')
    model = Sequential()
    model.add(Embedding(max_features, maxSequenceLength))
    model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    # обучаем
    batch_size = 64
    epochs = 8

    print('Тренируем модель...')
    history = model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, y_test))

    #сравниваем результаты
    score = model.evaluate(X_test, y_test,
                           batch_size=batch_size, verbose=1)

    print()
    print(u'Оценка теста: {}'.format(score[0]))
    print(u'Оценка точности модели: {}'.format(score[1]))

    #эффективность обучения
    Graphs.plot_efficiency(epochs, history)

    text_labels = encoder.classes_

    for i in range(20):
        prediction = model.predict(np.array([X_test[i]]))
        predicted_label = text_labels[np.argmax(prediction)]
        print('====================================')
        print('Правильная категория: {}'.format(y_test[i]))
        print("Определенная моделью категория: {}".format(predicted_label))


    y_softmax = model.predict(X_test)

    y_test_1d = []
    y_pred_1d = []

    for i in range(len(y_test)):
        probs = y_test[i]
        index_arr = np.nonzero(probs)
        one_hot_index = index_arr[0].item(0)
        y_test_1d.append(one_hot_index)

    for i in range(0, len(y_softmax)):
        probs = y_softmax[i]
        predicted_index = np.argmax(probs)
        y_pred_1d.append(predicted_index)

    text_labels = encoder.classes_
    cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
    plt.figure(figsize=(48,40))
    Graphs.plot_confusion_matrix(cnf_matrix, classes=text_labels, title="Confusion matrix")
    model.save('')
