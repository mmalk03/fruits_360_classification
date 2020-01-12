from src import fruit_clf
from src.fruit_clf import defaults
from src.my_inceptionv3 import MyInceptionV3
from src.my_resnet50 import MyResNet50
from src.my_vgg16 import MyVgg16


def get_network(model_name):
    return {
        'vgg16': MyVgg16(),
        'resnet50': MyResNet50(),
        'inceptionv3': MyInceptionV3()
    }[model_name]


my_vgg16 = get_network('vgg16')
my_resnet50 = get_network('resnet50')
my_inceptionv3 = get_network('inceptionv3')

fruit_clf.train(my_vgg16, defaults['train_data_dir'], defaults['val_data_dir'], defaults['epochs'], defaults['batch_size'])
fruit_clf.evaluate(my_vgg16, defaults['val_data_dir'])
fruit_clf.show_plot_history('histories/vgg16_history')
fruit_clf.show_comparison_2_histories('histories/vgg16_history_1', 'histories/vgg16_history_2')
fruit_clf.predict(my_vgg16, defaults['image_path'])
