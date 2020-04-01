import numpy as np
import itertools
import matplotlib.pyplot as plt


class Graphs:
    """
    Build graphs of results
    """

    @staticmethod
    def plot_confusion_matrix(cm, classes,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues, normalize=True):
        """
        Prints and plots the confusion matrix

        :param cm:
        :param classes: Text labels
        :param title: Name of graph
        :param cmap:
        :param normalize: Normalization can be applied by setting `normalize=True`
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
        plt.show()


    @staticmethod
    def plot_efficiency(epochs, history):
        """
        Prints and plots the loss and accuracy

        :param epochs: Model's epochs
        :param history: Models's history
        """

        plt.style.use('ggplot')
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
