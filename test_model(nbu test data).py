import pandas as pd
import numpy as np
import way
import cleaner.TextCleaner as tc
import preparer.TextPreparer as tp
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing.text import Tokenizer


file = 'data/letters1.txt'
first = tc.CleanText.open_file(file)
second = tc.CleanText.prepare_text(first)
n = len(second)

df = pd.read_pickle(way.pickle)
df = df.sample(frac=1).reset_index(drop=True)
categories = df.category_code
descriptions = df.description

tokenizer, textSequences = tp.PrepareText.tokenizer(descriptions)
X_train, y_train, X_test, y_test = tp.PrepareText.load_data_from_arrays(descriptions, categories, train_test_split=0.8)

total_unique_words, maxSequenceLength = tp.PrepareText.max_count(descriptions, tokenizer)
vocab_size = round(total_unique_words/10)

encoder, num_classes = tp.PrepareText.num_classes(y_train, y_test)

X_train, X_test, y_train, y_test = tp.PrepareText.transform_sets(vocab_size, descriptions ,X_train, X_test, y_train, y_test, maxSequenceLength, num_classes)

X_test_nbu = second.description

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_test_nbu.tolist())

X_test1 = tokenizer.texts_to_sequences(X_test_nbu.tolist())
X_test1 = sequence.pad_sequences(X_test1, maxlen=maxSequenceLength)

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
batch_size = 16
epochs = 11

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

model.save('model/save/model(nbu test data).h5')

text_labels = encoder.classes_

for i in range(n):
    prediction = model.predict(np.array([X_test1[i]]))
    predicted_label = text_labels[np.argmax(prediction)]
    dict_ans = {
        0: 'Positive',
        1: 'Negative',
        2: 'Hotline',
        3: 'Hooligan',
        4: 'Offer',
        5: 'SiteAndCoins'
    }
    if predicted_label in dict_ans:
        predicted_label = dict_ans[predicted_label]
    print('========================================')
    print("Определенная моделью категория: {}".format(predicted_label))
