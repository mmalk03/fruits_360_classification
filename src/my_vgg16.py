import numpy as np
from keras.applications import VGG16
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix

from src import util


class MyVgg16:
    base_model_plot_path = 'plots/vgg16_model_plot.png'
    top_model_plot_path = 'plots/vgg16_top_model_plot.png'
    train_bottleneck_features_path = 'weights/vgg16_bottleneck_features_train.npy'
    val_bottleneck_features_path = 'weights/vgg16_bottleneck_features_validation.npy'
    model_path = 'weights/vgg16_model.h5'
    filter_dir = 'plots/filters/vgg16'
    history_path = 'histories/vgg16_history'

    defaults = {
        'img_width': 224,
        'img_height': 224,
        'batch_size': 16,
        'epochs': 30
    }

    img_width = 0
    img_height = 0

    def __init__(self, img_width=defaults['img_width'], img_height=defaults['img_height']):
        self.img_width = img_width
        self.img_height = img_height

    def train(self, train_data_dir, val_data_dir, epochs=defaults['epochs'], batch_size=defaults['batch_size']):
        self.save_bottleneck_features(train_data_dir, val_data_dir, batch_size)
        return self.train_top_model(train_data_dir, val_data_dir, epochs, batch_size)

    def save_bottleneck_features(self, train_data_dir, val_data_dir, batch_size):
        train_generator = util.get_generator(train_data_dir, self.img_width, self.img_height, batch_size)
        val_generator = util.get_generator(val_data_dir, self.img_width, self.img_height, batch_size)

        model = self.get_base_model()
        util.save_model_plot(self.base_model_plot_path, model)

        train_bottleneck_features = model.predict_generator(
            train_generator, len(train_generator.filenames) // batch_size)
        util.save_bottleneck_features(self.train_bottleneck_features_path, train_bottleneck_features)

        val_bottleneck_features = model.predict_generator(
            val_generator, len(val_generator.filenames) // batch_size)
        util.save_bottleneck_features(self.val_bottleneck_features_path, val_bottleneck_features)

    def train_top_model(self, train_data_dir, val_data_dir, epochs=defaults['epochs'],
                        batch_size=defaults['batch_size']):
        train_generator = util.get_generator(train_data_dir, self.img_width, self.img_height, batch_size)
        val_generator = util.get_generator(val_data_dir, self.img_width, self.img_height, batch_size)

        train_data = util.load_bottleneck_features(self.train_bottleneck_features_path)
        val_data = util.load_bottleneck_features(self.val_bottleneck_features_path)

        num_classes = train_generator.num_classes

        train_labels = train_generator.classes
        train_labels = to_categorical(train_labels, num_classes=num_classes)
        val_labels = val_generator.classes
        val_labels = to_categorical(val_labels, num_classes=num_classes)

        model = self.get_top_model(train_data.shape[1:], num_classes)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        util.save_model_plot(self.top_model_plot_path, model)

        history = model.fit(train_data, train_labels,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(val_data, val_labels))
        model.save(self.model_path)
        util.save_history(self.history_path, history)
        util.eval_model_loss_acc(model, val_data, val_labels, batch_size)

    def evaluate(self, data_dir, batch_size=defaults['batch_size']):
        test_generator = util.get_generator(data_dir, self.img_width, self.img_height, batch_size)

        base_model = self.get_base_model()
        test_bottleneck_features = base_model.predict_generator(
            test_generator, len(test_generator.filenames) // batch_size)

        num_classes = test_generator.num_classes

        test_labels = test_generator.classes
        test_labels = to_categorical(test_labels, num_classes=num_classes)

        top_model = load_model(self.model_path)
        test_loss, test_acc = top_model.evaluate(test_bottleneck_features, test_labels, batch_size=batch_size)

        print('Test accuracy: ', test_acc)
        print('Test loss: ', test_loss)
        return test_acc, test_loss

    def make_prediction(self, path):
        image = util.load_image(path, self.img_width, self.img_height)

        base_model = self.get_base_model()
        bottleneck_prediction = base_model.predict(image)
        top_model = load_model(self.model_path)
        return top_model.predict_classes(bottleneck_prediction)

    def get_history(self):
        return util.load_history(self.history_path)

    def get_confusion_matrix(self, directory, batch_size=defaults['batch_size']):
        validation_data = util.load_bottleneck_features(self.val_bottleneck_features_path)
        val_generator = util.get_generator(directory, self.img_width, self.img_height, batch_size)
        train_labels = val_generator.classes
        top_model = load_model(self.model_path)
        predicted_labels = top_model.predict_classes(validation_data)
        return confusion_matrix(train_labels, predicted_labels)

    def get_wrong_predictions(self, directory, batch_size=defaults['batch_size']):
        validation_data = util.load_bottleneck_features(self.val_bottleneck_features_path)
        val_generator = util.get_generator(directory, self.img_width, self.img_height, batch_size)
        train_labels = val_generator.classes
        top_model = load_model(self.model_path)
        return np.nonzero(top_model.predict_classes(validation_data).reshape((-1,)) != train_labels)

    @staticmethod
    def get_top_model(input_shape, num_classes):
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        return model

    def get_base_model(self):
        return VGG16(include_top=False, weights='imagenet', input_shape=(self.img_width, self.img_height, 3))

    def visualize_filters(self):
        model = self.get_base_model()
        for i in range(64):
            util.vis_filter(self.filter_dir, self.img_width, self.img_height,
                            model, 'block1_conv1', i)
