import itertools
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import way
import cleaner.TextCleaner as tc
import preparer.TextPreparer as tp
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt


file = 'data/data_nbu.csv'
first = pd.read_csv(file)
second = tc.CleanText.prepare_text(first)
second['category_code'] = second.category.apply(lambda x: 5 if x == 'SiteAndCoins' else 2)
cat_code = second.category_code
desc = second.description

n = len(cat_code)

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

model.save('model/save/model(nbu test data with answ).h5')

text_labels = encoder.classes_

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(desc.tolist())

X_test1 = tokenizer.texts_to_sequences(desc.tolist())
X_test1 = sequence.pad_sequences(X_test1, maxlen=maxSequenceLength)

match = 0
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

    if cat_code[i] in dict_ans:
        cat = dict_ans[cat_code[i]]

    if cat == predicted_label:
        match += 1

    print('========================================')
    print("Определенная моделью категория: {}".format(predicted_label))
    print('Правильная категория: {}'.format(cat))
print('Совпадений: ', match)

plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), history.history['loss'], label='train_loss')
plt.plot(np.arange(0, N), history.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, N), history.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, N), history.history['val_accuracy'], label='val_acc')
plt.title('Эффективность обучения')
plt.xlabel('Повторения #')
plt.ylabel('Ошибки')
plt.legend(loc='lower left')
plt.show()

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, normalize=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=22)
    plt.yticks(tick_marks, classes, fontsize=22)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('Правильная категория', fontsize=25)
    plt.xlabel('Определенная моделью категория', fontsize=25)


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
plot_confusion_matrix(cnf_matrix, classes=text_labels, title='Confusion matrix')
plt.show()
