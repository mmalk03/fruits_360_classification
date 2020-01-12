#!/usr/bin/env python
import itertools
import os
from random import randint

import matplotlib.pyplot as plt
import numpy as np

from src import util

defaults = {
    "epochs": 30,
    "batch_size": 16,
    "train_data_dir": "fruits-360/data/train",
    "val_data_dir": "fruits-360/data/validation",
    "image_path": "banana.jpg"
}


def train(network, train_data_dir, val_data_dir, epochs, batch_size):
    """
    trains given network, using provided arguments
    :param network: object MyVGG16, MyResnet50 or MyInceptionV3
    :param train_data_dir:
    :param val_data_dir:
    :param epochs:
    :param batch_size:
    """
    network.train(train_data_dir, val_data_dir, epochs, batch_size)


def evaluate(network, eval_data_dir):
    """
    :param network: object MyVGG16, MyResnet50 or MyInceptionV3
    :param eval_data_dir: directory containing evaluation data, e.g. fruits-360/data/validation
    :return: pair of accuracy and loss
    """
    accuracy, loss = network.evaluate(eval_data_dir)
    return accuracy, loss


def predict(network, image_path):
    """
    :param network: object MyVGG16, MyResnet50 or MyInceptionV3
    :param image_path: path to image which will be classified
    :return: name of predicted object as string
    """
    predicted_class = network.make_prediction(image_path)

    class_dictionary = util.get_class_dictionary(defaults['train_data_dir'])
    in_id = predicted_class[0]
    inv_map = {v: k for k, v in class_dictionary.items()}

    label = inv_map[in_id]
    return label
    # print("Image ID: {}, Label: {}".format(in_id, label))
    # original_image = cv2.imread(image_path)
    # cv2.putText(
    #   original_image, "Predicted: {}".format(label), (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255),2)
    # cv2.imshow("Classification", original_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def show_plot_history(history_path):
    """
    shows plot of training history
    :param history_path: path to saved Keras history object, e.g. 'histories/vgg16_history'
    """
    hist = util.load_history(history_path)
    lt = 'train'
    lv = 'validation'

    plt.figure()
    plt.plot(hist['acc'], label=lt, color='firebrick')
    plt.plot(hist['val_acc'], label=lv, color='orangered')
    plt.title('Accuracy change during training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(hist['loss'], label=lt, color='firebrick')
    plt.plot(hist['val_loss'], label=lv, color='orangered')
    plt.title('Loss change during training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def show_comparison_2_histories(hist1_path, hist2_path):
    """
    shows comparison of 2 training histories
    :param hist1_path: path to saved Keras history object, e.g. 'histories/vgg16_history'
    :param hist2_path: path to saved Keras history object, e.g. 'histories/vgg16_history'
    :return:
    """
    l1t = '1. train'
    l1v = '1. validation'
    l2t = '2. train'
    l2v = '2. validation'

    hist1 = util.load_history(hist1_path)
    hist2 = util.load_history(hist2_path)

    plt.figure()
    plt.plot(hist1['acc'], label=l1t, color='firebrick')
    plt.plot(hist1['val_acc'], label=l1v, color='orangered')
    plt.plot(hist2['acc'], label=l2t, color='springgreen')
    plt.plot(hist2['val_acc'], label=l2v, color='forestgreen')
    plt.title('Accuracy change during training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(hist1['loss'], label=l1t, color='firebrick')
    plt.plot(hist1['val_loss'], label=l1v, color='orangered')
    plt.plot(hist2['loss'], label=l2t, color='springgreen')
    plt.plot(hist2['val_loss'], label=l2v, color='forestgreen')
    plt.title('Loss change during training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(network, data_dir, save_path, normalize=False, title='Confusion matrix'):
    conf_mat = network.get_confusion_matrix(data_dir)
    if normalize:
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(conf_mat)

    class_dictionary = util.get_class_dictionary(data_dir)

    np.set_printoptions(precision=2)
    plt.figure()
    tick_marks = np.arange(len(class_dictionary))
    plt.title(title)
    plt.imshow(conf_mat, interpolation='none', cmap=plt.get_cmap('Blues'))
    plt.colorbar()
    plt.xticks(tick_marks, class_dictionary.keys(), rotation=20, fontsize=10)
    plt.yticks(tick_marks, class_dictionary.keys(), fontsize=10)

    fmt = '.2f' if normalize else 'd'
    thresh = conf_mat.max() / 2.
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, format(conf_mat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conf_mat[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def show_incorrect_predictions(network, data_dir):
    images = []
    for root, directories, filenames in os.walk(data_dir, topdown=True):
        directories.sort(reverse=False)
        for filename in filenames:
            images.append(os.path.join(root, filename))

    class_dictionary = util.get_class_dictionary(data_dir)
    incorrects = network.get_wrong_predictions(data_dir)

    for i in range(4):
        random_image_index = randint(0, len(incorrects))
        wrong_prediction = incorrects[random_image_index]
        wrong_image_index = wrong_prediction[0]
        true_label_index = wrong_prediction[1]
        pred_label_index = wrong_prediction[2]

        random_wrong_image = images[wrong_image_index]
        pred_label = [k for (k, v) in class_dictionary.items() if v == pred_label_index]
        true_label = [k for (k, v) in class_dictionary.items() if v == true_label_index]
        print('Predicted: ', pred_label)
        print('Path: ', random_wrong_image)
        print('Actual: ', true_label)
