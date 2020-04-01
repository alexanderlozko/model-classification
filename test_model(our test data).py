import pandas as pd
import numpy as np
import way
import preparer.TextPreparer as tp
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


df = pd.read_pickle(way.pickle)
df = df.sample(frac=1).reset_index(drop=True)
categories = df.category_code
descriptions = df.description
text = df.text

tokenizer, textSequences = tp.PrepareText.tokenizer(descriptions)
X_train, y_train, X_test, y_test = tp.PrepareText.load_data_from_arrays(descriptions, categories, train_test_split=0.8)

X_train_n, y_train_n, X_test_n,  y_test_n = tp.PrepareText.load_data_from_arrays(text, categories, train_test_split=0.8)

test_df = (pd.DataFrame({'text':list(X_test_n),
                       'category': list(y_test_n)})).to_csv('data/out_test_data.csv', index=False)
n = len(y_test_n)

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

model.save('model/save/model(our test data).h5')

text_labels = encoder.classes_

match = 0
for i in range(n):
    prediction = model.predict(np.array([X_test[i]]))
    predicted_label = text_labels[np.argmax(prediction)]
    dict_ans = {
        0:'Positive',
        1:'Negative',
        2:'Hotline',
        3:'Hooligan',
        4:'Offer',
        5:'SiteAndCoins'
    }
    if predicted_label in dict_ans:
        predicted_label = dict_ans[predicted_label]

    if list(y_test[i]).index(1) in dict_ans:
        cat = dict_ans[list(y_test[i]).index(1)]

    if cat == predicted_label:
        match += 1

    print('========================================')
    print("Определенная моделью категория: {}".format(predicted_label))
    print('Правильная категория: {}'.format(cat))
print('Совпадений: ', match)

plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Эффективность обучения")
plt.xlabel("Повторения #")
plt.ylabel("Ошибки")
plt.legend(loc="lower left")
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
plt.figure(figsize=(48, 40))
plot_confusion_matrix(cnf_matrix, classes=text_labels, title='Confusion matrix')
plt.show()
