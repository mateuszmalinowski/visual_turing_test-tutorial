"""
Different Visual Architectures.

Inspired by:
    baraldilorenzo vgg16 model for Keras
    MarcBS caffe to keras transformation

Author: Mateusz Malinowski
Email: mmalinow@mpi-inf.mpg.de
"""

from keras.models import Sequential

from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D

from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Reshape


###
# Functions
###
def imagenet_mean_preprocess_image_tensor_fun(x):
    """
    In:
        x - image tensor of size (#images, #channels, #dim1, #dim2)

    Out:
        image tensor x with subtracted imagenet mean
    """
    y = x 
    y[:,0,:,:] -= 103.939
    y[:,1,:,:] -= 116.779
    y[:,2,:,:] -= 123.68
    return y


def get_visual_features(
        data_provider,
        trainable_perception_name,
        train_or_test,
        image_names_list,
        parts_extractor,
        max_parts,
        perception,
        layer,
        second_layer,
        preprocess_image_tensor_fun
        ):
    """
    In:
        data_provider - data provider function
        train_or_test - training, validation, or test set
        image_names_list - list with names of images
        parts_extractor - name for the parts extractor
        max_parts - maximal number of parts if these are extracted
        perception - name of the perception model 
            if the perception is fixed (pre-trained)
        layer - name for the perception's layer
        second_layer - name for the second parception's layer
        trainable_perception_name - name for the perception 
            if the perception is not fixed
        preprocess_image_tensor_fun - image preprocessing function;
            only if trainable_perception_name is not 'none'

    Out:
        image features, or image tensor 
    """
    if trainable_perception_name == 'none':
        visual_features = data_provider['perception'](
                train_or_test=train_or_test,
                names_list=image_names_list,
                parts_extractor=parts_extractor,
                max_parts=max_parts,
                perception=perception,
                layer=layer,
                second_layer=second_layer)
    else:
        visual_features = preprocess_image_tensor_fun(
                data_provider['images'](
                    train_or_test=train_or_test,
                    names_list=image_names_list))
    return visual_features
 

###
# Abstract building visual models.
###
class AbstractVisualModel():
    """
    Abstract class to build visual models.
    """
    def __init__(self, visual_dim, weights_path=None):
        """
        In:
            visual_dim - dimensionality of the input space;
                it can be a tuple, or a scalar
            weights_path - path to the weights to load, by default None
        """
        self._weights_path = weights_path
        self._visual_dim = visual_dim 

    def create(self):
        """
        Creates a model.

        Out:
            model
        """
        raise NotImplementedError()

    def get_dimensionality(self):
        """
        Out:
            Returns an output dimensionality of this layer.
        """
        raise NotImplementedError()


###
# Concrete building visual models.
###
class SequentialVisualModelEmpty(AbstractVisualModel):
    """
    Empty visual model. No model. 
    """
    def create(self):
        model = Sequential()
        model.add(Reshape(
            input_shape=(self._visual_dim,),
            dims=(self._visual_dim,)))
        return model

    def get_dimensionality(self):
        return self._visual_dim


class SequentialVisualModelVGG16(AbstractVisualModel):
    """
    Sequential visual model.

    VGG16
    """
    def create(self):
        model = Sequential()

        model.add(ZeroPadding2D((1,1), input_shape=self._visual_dim))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))

        self._model_output_dim = 4096
        model.add(Dense(self._model_output_dim, activation='relu'))
        model.add(Dropout(0.5))

        #model.add(Dense(1000, activation='softmax'))

        if self._weights_path:
            model.load_weights(self._weights_path)
        return model

    def get_dimensionality(self):
        return self._model_output_dim


class SequentialVisualModelVeryShallowCNN(AbstractVisualModel):
    """
    Sequential visual model.

    Small CNN.
    """
    def create(self):
        model = Sequential()

        model.add(ZeroPadding2D((1,1), input_shape=self._visual_dim))
        model.add(Convolution2D(64, 3, 3, activation='relu'))

        model.add(Flatten())
        self._model_output_dim = 4096
        model.add(Dense(self._model_output_dim, activation='relu'))
        model.add(Dropout(0.5))

        if self._weights_path:
            model.load_weights(self._weights_path)
        return model

    def get_dimensionality(self):
        return self._model_output_dim


###
# Selector
###
select_sequential_visual_model = {
    'none':SequentialVisualModelEmpty, 
    'vgg16':SequentialVisualModelVGG16,
    'very_shallow_cnn':SequentialVisualModelVeryShallowCNN
    }
