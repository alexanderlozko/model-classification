from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import cleaner.TextCleaner as tc
import preparer.TextPreparer as tp
import way
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.preprocessing import sequence
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import GridSearchCV


file = '../data/data_nbu.csv'
first = pd.read_csv(file)
second = tc.CleanText.prepare_text(first)
second['category_code'] = second.category.apply(lambda x: 5 if x == 'SiteAndCoins' else 2)
cat_code = second.category_code
desc = second.description

n = len(cat_code)

df = pd.read_pickle(way.pickle1)
df = df.sample(frac=1).reset_index(drop=True)
categories = df.category_code
descriptions = df.description

tokenizer, textSequences = tp.PrepareText.tokenizer(descriptions)

X_train, y_train, X_test, y_test = tp.PrepareText.load_data_from_arrays(descriptions, categories, train_test_split=0.8)

total_unique_words, maxSequenceLength = tp.PrepareText.max_count(descriptions, tokenizer)
vocab_size = round(total_unique_words/10)


encoder, num_classes = tp.PrepareText.num_classes(y_train, y_test)

Y_train = y_train
Y_test = y_test
X_train, X_test, y_train, y_test = tp.PrepareText.transform_sets(vocab_size, descriptions ,X_train, X_test, y_train, y_test, maxSequenceLength, num_classes)

classifier = RandomForestClassifier(criterion='gini', max_depth=21, n_estimators=1)

# parametrs = {'criterion':['entropy', 'gini'],
#              'max_depth': range(1,50),
#              'n_estimators': range(1,10)}
# grid = GridSearchCV(classifier, parametrs, cv=5)
# grid.fit(X_train, y_train)
# print(grid)
# print(grid.best_params_)
# print(grid.best_score_)

classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

report = classification_report(Y_test, y_pred)
print(report)

matrix = confusion_matrix(Y_test, y_pred)
print(matrix)

text_labels = encoder.classes_

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(desc.tolist())

X_test1 = tokenizer.texts_to_sequences(desc.tolist())
X_test1 = sequence.pad_sequences(X_test1, maxlen=maxSequenceLength)

match = 0
for i in range(n):
    prediction = classifier.predict(np.array([X_test1[i]]))
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

    #NBU DATA
    if cat_code[i] in dict_ans:
        cat = dict_ans[cat_code[i]]
    #OUR DATA
    # if list(y_test[i]).index(1) in dict_ans:
    #     cat = dict_ans[list(y_test[i]).index(1)]

    if cat == predicted_label:
        match += 1

    print('========================================')
    print("Определенная моделью категория: {}".format(predicted_label))
    print('Правильная категория: {}'.format(cat))
print('Совпадений: ', match)
