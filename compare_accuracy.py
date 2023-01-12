from matplotlib import pyplot as plt
import numpy as np


def draw_result(lst_iter, lst_acc, title):
    plt.plot(lst_iter, lst_acc, '-r', label='accuracy')

    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    plt.title(title)
    plt.savefig(title+".png")  # should before show method

    plt.show()

if __name__ == '__main__':
    acc_2layer = np.load("accuracy_4layer.npy")
    acc_original = np.load("accuracy_original.npy")
    
    length = min(len(acc_2layer), len(acc_original))
    
    acc_iter = range(length)
    plt.plot(acc_iter, acc_2layer, '-r', label='multi-layer accuracy')
    plt.plot(acc_iter, acc_original, '-b', label='original accuracy')

    plt.xlabel("n iteration")
    plt.ylabel("accuracy")
    plt.legend(loc='lower right')
    title = "compare accuracy on training data_1"
    plt.title(title)
    plt.savefig(title+".png")  # should before show method

    # plt.show()
