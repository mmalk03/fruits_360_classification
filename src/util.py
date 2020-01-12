import pickle
from pathlib import Path

import numpy as np
import scipy
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import plot_model


def get_class_dictionary(directory):
    return ImageDataGenerator().flow_from_directory(directory).class_indices


def get_generator(directory, img_width, img_height, batch_size):
    data_generator = ImageDataGenerator(rescale=1. / 255)
    return data_generator.flow_from_directory(
        directory,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)


def get_augmented_generator(directory, img_width, img_height, batch_size):
    data_generator = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    return data_generator.flow_from_directory(
        directory,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)


def get_categorical_generator(directory, img_width, img_height, batch_size):
    data_generator = ImageDataGenerator(rescale=1. / 255)
    return data_generator.flow_from_directory(
        directory,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')


def save_bottleneck_features(path, bottleneck_features):
    np.save(open(path, 'wb'), bottleneck_features)


def load_bottleneck_features(path):
    return np.load(open(path, 'rb'))


def save_model_plot(path, model):
    plot_model(model, to_file=path, show_shapes=True, show_layer_names=True)
    print(model.summary())


def eval_model_loss_acc(model, validation_data, validation_labels, batch_size):
    (eval_loss, eval_accuracy) = model.evaluate(
        validation_data, validation_labels, batch_size=batch_size, verbose=1)
    print("[INFO] Accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))


def load_model_weights(path, model):
    my_file = Path(path)
    if my_file.is_file():
        model.load_weights(path)


def save_history(path, history):
    with open(path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


def load_history(path):
    filename = open(path, "rb")
    history = pickle.load(filename)
    filename.close()
    return history


def load_image(image_path, width, height):
    image = load_img(image_path, target_size=(width, height))
    image = img_to_array(image)
    image = image / 255
    image = np.expand_dims(image, axis=0)
    return image


def vis_filter(save_dir, img_width, img_height, model, layer_name, filter_index):
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    input_img = layer_dict[layer_name].input
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, input_img)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([input_img], [loss, grads])
    step = 1.

    input_img_data = (np.random.random((1, img_width, img_height, 3)) - 0.5) * 20 + 128
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    img = deprocess_img(img)
    scipy.misc.toimage(img, cmin=0, cmax=255).save(save_dir + '/%s_filter_%d.png' % (layer_name, filter_index))


def deprocess_img(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x
