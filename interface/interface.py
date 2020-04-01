import tkinter as tk
from tkinter import messagebox
from tkinter.ttk import Radiobutton
from tkinter import filedialog as fd
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import cleaner.TextCleaner as tc
import preparer.TextPreparer as tp
import way
from model.Model_new import ModelFile


class ModelInterface:
    """
    Calls up functions depending on the user choosing a button on the initial window
    """

    @staticmethod
    def create_window():
        """
        Creates a message box
        """

        window = tk.Toplevel(root)
        program_message = tk.Label(window, text='Введіть звернення в поле нижче для визначення категорії\n'
                                                '(лише українською або російською мовою)\n'
                                                'класификація може зайняти декілька хвилин')
        window.title('Визначення категорії')
        user_message = tk.Text(window)
        define = tk.Button(window, text='Визначити категорію')
        input_field = tk.Label(window, bg='white', fg='black')
        scroll = tk.Scrollbar(window, command=user_message.yview)

        def yes_clicked():
            """
            If the category is defined correctly
            """

            messagebox.showinfo(message='Дякуємо за звернення!')
            window.destroy()

        def no_clicked():
            """
            If the category is not defined correctly
            """

            choose = tk.Toplevel(window)
            program_message = tk.Label(choose,
                                       text='Вкажіть, до якої категорії належить звернення,'
                                            '\n та ми продовжимо навчання моделі')
            var = tk.IntVar()
            var.set(0)
            category0 = Radiobutton(choose, text='Позитивний відгук', variable=var, value=0)
            category1 = Radiobutton(choose, text='Негативний відгук', variable=var, value=1)
            category2 = Radiobutton(choose, text='Звернення на внутрішній департамент', variable=var, value=2)
            category3 = Radiobutton(choose, text='Хуліганське звернення', variable=var, value=3)
            category4 = Radiobutton(choose, text='Пропозиція щодо вдосконалення роботи', variable=var, value=4)
            category5 = Radiobutton(choose, text='Звернення стосовно сайту купівлі монет', variable=var, value=5)
            text = user_message.get('1.0', 'end-1c')

            def learn():
                """
                Continues model training on the entered message
                """

                messagebox.showinfo(message='Дякуємо за звернення!')
                window.destroy()

                if var.get() == 0:
                    category = 'Positive'
                elif var.get() == 1:
                    category = 'Negative'
                elif var.get() == 2:
                    category = 'Hotline'
                elif var.get() == 3:
                    category = 'Hooligan'
                elif var.get() == 4:
                    category = 'Offer'
                elif var.get() == 5:
                    category = 'SiteAndCoins'

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

                cat = '../data/category_with_received_data.csv'
                with open(cat, 'a') as f:
                    f.write('"{}",{}\n'.format(text, category))

                # y_train = keras.utils.to_categorical(y_train, num_classes)
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

                model.fit(X_train, np.array(y_train), batch_size=1,
                          epochs=1)



            choose_button = tk.Button(choose, text='Підтвердити', command=learn)

            program_message.grid()
            category0.grid()
            category1.grid()
            category2.grid()
            category3.grid()
            category4.grid()
            category5.grid()
            choose_button.grid()
            choose.mainloop()

        def define_category(event):
            """
            Defining the category of the entered message
            """

            text = user_message.get('1.0', 'end-1c')
            if text == '':
                messagebox.showerror('Помилка', 'Введіть текст звернення')

            else:
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
                    if predicted_label == 'Positive':
                        predicted_label = 'Позитивний відгук'
                    elif predicted_label == 'Negative':
                        predicted_label = 'Негативний відгук'
                    elif predicted_label == 'Hotline':
                        predicted_label = 'Звернення на внутрішній департамент'
                    elif predicted_label == 'Hooligan':
                        predicted_label = 'Хуліганське звернення'
                    elif predicted_label == 'Offer':
                        predicted_label = 'Пропозиція щодо вдосконалення роботи'
                    elif predicted_label == 'SiteAndCoins':
                        predicted_label = 'Звернення стосовно сайту купівлі монет'
                input_field['text'] = predicted_label

                yes = tk.Button(window, text='Правильно', command=yes_clicked)
                no = tk.Button(window, text='Неправильно', command=no_clicked)
                yes.pack(side='left', padx=35, pady=35)
                no.pack(side='right', padx=35, pady=35)

        define.bind('<Button-1>', define_category)

        program_message.pack()
        scroll.pack(side='right', fill='y')
        user_message.config(yscrollcommand=scroll.set)
        user_message.pack()
        define.pack()
        input_field.pack()
        window.mainloop()

    @staticmethod
    def create_file_window():
        """
        Creates a file download window for training
        """

        window = tk.Toplevel(root)
        window.title('Продовження навчання моделі')
        program_message = tk.Label(window, text='Оберіть файл', font=("Arial Bold", 14))
        program_message1 = tk.Label(window, text='Файл повинен бути формату .csv '
                                                 'та містити дані, збережені наступним чином:', font=("Arial Bold", 12))
        img = Image.open('csv_example.png')
        render = ImageTk.PhotoImage(img)
        initil = tk.Label(window, image=render)
        initil.image = render

        def insertText():
            """
            File selection
            """

            file_name = fd.askopenfilename()
            format = file_name.split('.')[-1]
            if format != 'csv':
                messagebox.showerror('Помилка', 'Файл повинен бути формату .csv')
            else:
                print_file_name = file_name.split('/')[-1]
                print_file_name = tk.Label(window, text='Обраний файл: ' + print_file_name)

                def file_learn():
                    """
                    Continuation of model training on data from the selected file
                    """

                    messagebox.showinfo(message='Дякуємо за звернення!\n'
                                                'Ми продовжимо навчання моделі на даних з файлу')
                    window.destroy()

                    model_class = ModelFile('../model/save/model(nbu test data with answ).h5', file_name, '../data/category_with_received_data.csv', 64, 1)
                    model_class.read()
                    model_class.clean_and_prapare()
                    total_unique_words = model_class.data_to_vect()
                    model_class.vocab_size = total_unique_words
                    model_class.data_final()
                    model_class.train_model()

                learn_button = tk.Button(window, text='Продовжити навчання моделі', command=file_learn)
                print_file_name.grid()
                learn_button.grid()

        open_button = tk.Button(window, text='Обрати файл', command=insertText)
        program_message.grid()
        program_message1.grid()
        initil.grid()
        open_button.grid()
        window.mainloop()

root = tk.Tk()
w = root.winfo_screenwidth()
h = root.winfo_screenheight()
w = w//2
h = h//2
w = w - 200
h = h - 200
root.geometry('400x200+{}+{}'.format(w, h))
root.title('Класифікатор звернень громадян')

program_message = tk.Label(root, text='Ви знаходитесь в класификаторі звернень громадян', font=('Arial Bold', 14))
define_category = tk.Button(root, text='Визначити категорію', command=ModelInterface.create_window)
continue_training = tk.Button(root, text="Продовжити навчання моделі", command=ModelInterface.create_file_window)

program_message.pack()
define_category.pack()
continue_training.pack()
root.mainloop()
