import pandas as pd
import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import cleaner.TextCleaner as tc
import preparer.TextPreparer as tp


text = str(input('TEXT: '))

text_list = list()
for x in range(6):
    text_list.append(text)
model = load_model('../model/save/model(our test data).h5')
df = pd.DataFrame({'text': text_list})
X_train = df.text
cat = '../data/category_with_received_data.csv'
all = pd.read_csv(cat)

cat = tc.CleanText.clean_category(all)

descriptions = tc.CleanText.prepare_text(all).description

maxSequenceLength, vocab_size, encoder, num_classes = tp.PrepareText.parameters(cat)

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(descriptions)

X_train = tokenizer.texts_to_sequences(X_train)

X_train = sequence.pad_sequences(X_train, maxlen=maxSequenceLength)

model.predict(X_train)

text_labels = encoder.classes_

prediction = model.predict([X_train])
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

ans = input('Модель правильно определила категорию?(y/n): ')
if ans == 'n':
    category = str(input('CATEGORY: '))

    cat = '../data/category_with_received_data.csv'
    with open(cat, 'a') as f:
        f.write('"{}",{}\n'.format(text, category))

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
        y_train.append(dict_changes[category])

    model.fit(X_train, np.array(y_train),
              batch_size=1, epochs=1)
