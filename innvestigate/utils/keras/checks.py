# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################

import tensorflow.keras as keras
import inspect
# import keras.engine.topology
# import tensorflowkeras.layers
# import keras.layers.advanced_activations
# import keras.layers.convolutional
# import keras.layers.convolutional_recurrent
# import keras.layers.core
# import keras.layers.cudnn_recurrent
# import keras.layers.embeddings
# import keras.layers.local
# import keras.layers.noise
# import keras.layers.normalization
# import keras.layers.pooling
# import keras.layers.recurrent
# import keras.layers.wrappers
# import keras.legacy.layers


# Prevents circular imports.
def get_kgraph():
    from . import graph as kgraph
    return kgraph


__all__ = [
    "get_current_layers",
    "get_known_layers",
    "get_activation_search_safe_layers",

    "contains_activation",
    "contains_kernel",
    "only_relu_activation",
    "is_network",
    "is_convnet_layer",
    "is_relu_convnet_layer",
    "is_average_pooling",
    "is_max_pooling",
    "is_input_layer",
    "is_batch_normalization_layer",
]


###############################################################################
###############################################################################
###############################################################################


def get_current_layers():
    """
    Returns a list of currently available layers in Keras.
    """
    class_set = set([(getattr(keras.layers, name), name)
                     for name in dir(keras.layers)
                     if (inspect.isclass(getattr(keras.layers, name)) and
                         issubclass(getattr(keras.layers, name),
                                        keras.layers.Layer))])
    return [x[1] for x in sorted((str(x[0]), x[1]) for x in class_set)]


def get_known_layers():
    """
    Returns a list of keras layer we are aware of.
    """

    # Inside function to not break import if Keras changes.
    KNOWN_LAYERS = (
        keras.layers.InputLayer,
        keras.layers.ELU,
        keras.layers.LeakyReLU,
        keras.layers.PReLU,
        keras.layers.Softmax,
        keras.layers.ThresholdedReLU,
        keras.layers.Conv1D,
        keras.layers.Conv2D,
        keras.layers.Conv2DTranspose,
        keras.layers.Conv3D,
        keras.layers.Conv3DTranspose,
        keras.layers.Cropping1D,
        keras.layers.Cropping2D,
        keras.layers.Cropping3D,
        keras.layers.SeparableConv1D,
        keras.layers.SeparableConv2D,
        keras.layers.UpSampling1D,
        keras.layers.UpSampling2D,
        keras.layers.UpSampling3D,
        keras.layers.ZeroPadding1D,
        keras.layers.ZeroPadding2D,
        keras.layers.ZeroPadding3D,
        keras.layers.ConvLSTM2D,
        keras.layers.Activation,
        keras.layers.ActivityRegularization,
        keras.layers.Dense,
        keras.layers.Dropout,
        keras.layers.Flatten,
        keras.layers.Lambda,
        keras.layers.Masking,
        keras.layers.Permute,
        keras.layers.RepeatVector,
        keras.layers.Reshape,
        keras.layers.SpatialDropout1D,
        keras.layers.SpatialDropout2D,
        keras.layers.SpatialDropout3D,
        keras.layers,
        keras.layers.Embedding,
        keras.layers.LocallyConnected1D,
        keras.layers.LocallyConnected2D,
        keras.layers.Add,
        keras.layers.Average,
        keras.layers.Concatenate,
        keras.layers.Dot,
        keras.layers.Maximum,
        keras.layers.Minimum,
        keras.layers.Multiply,
        keras.layers.Subtract,
        keras.layers.AlphaDropout,
        keras.layers.GaussianDropout,
        keras.layers.GaussianNoise,
        keras.layers.BatchNormalization,
        keras.layers.AveragePooling1D,
        keras.layers.AveragePooling2D,
        keras.layers.AveragePooling3D,
        keras.layers.GlobalAveragePooling1D,
        keras.layers.GlobalAveragePooling2D,
        keras.layers.GlobalAveragePooling3D,
        keras.layers.GlobalMaxPooling1D,
        keras.layers.GlobalMaxPooling2D,
        keras.layers.GlobalMaxPooling3D,
        keras.layers.MaxPooling1D,
        keras.layers.MaxPooling2D,
        keras.layers.MaxPooling3D,
        keras.layers.GRU,
        keras.layers.GRUCell,
        keras.layers.LSTM,
        keras.layers.LSTMCell,
        keras.layers.RNN,
        keras.layers.SimpleRNN,
        keras.layers.SimpleRNNCell,
        keras.layers.StackedRNNCells,
        keras.layers.Bidirectional,
        keras.layers.TimeDistributed,
        keras.layers.Wrapper,
        keras.layers.Highway,
        keras.layers.MaxoutDense,
        keras.layers.Merge,
        keras.layers.Recurrent,
    )
    return KNOWN_LAYERS


def get_activation_search_safe_layers():
    """
    Returns a list of keras layer that we can walk along
    in an activation search.
    """

    # Inside function to not break import if Keras changes.
    ACTIVATION_SEARCH_SAFE_LAYERS = (
        keras.layers.ELU,
        keras.layers.LeakyReLU,
        keras.layers.PReLU,
        keras.layers.Softmax,
        keras.layers.ThresholdedReLU,
        keras.layers.Activation,
        keras.layers.ActivityRegularization,
        keras.layers.Dropout,
        keras.layers.Flatten,
        keras.layers.Reshape,
        keras.layers.Add,
        keras.layers.GaussianNoise,
        keras.layers.BatchNormalization,
    )
    return ACTIVATION_SEARCH_SAFE_LAYERS


###############################################################################
###############################################################################
###############################################################################


def contains_activation(layer, activation=None):
    """
    Check whether the layer contains an activation function.
    activation is None then we only check if layer can contain an activation.
    """

    # todo: add test and check this more throughroughly.
    # rely on Keras convention.
    if hasattr(layer, "activation"):
        if activation is not None:
            return layer.activation == keras.activations.get(activation)
        else:
            return True
    elif isinstance(layer, keras.layers.ReLU):
        if activation is not None:
            return (keras.activations.get("relu") ==
                    keras.activations.get(activation))
        else:
            return True
    elif isinstance(layer, (
            keras.layers.ELU,
            keras.layers.LeakyReLU,
            keras.layers.PReLU,
            keras.layers.Softmax,
            keras.layers.ThresholdedReLU)):
        if activation is not None:
            raise Exception("Cannot detect activation type.")
        else:
            return True
    else:
        return False



def contains_any_activation(layer):
    """Check whether layer contains any activation or is activation layer.

    :param layer: Keras layer to check
    :type layer: Layer
    :return: True if activation was found.
    :rtype: bool
    """
    activation_layers = (
        keras.layers.ReLU,
        keras.layers.ELU,
        keras.layers.LeakyReLU,
        keras.layers.PReLU,
        keras.layers.Softmax,
        keras.layers.ThresholdedReLU,
    )
    return hasattr(layer, "activation") or isinstance(layer, activation_layers)



def contains_kernel(layer):
    """
    Check whether the layer contains a kernel.
    """

    # TODO: add test and check this more throughroughly.
    # rely on Keras convention.
    if hasattr(layer, "kernel") or hasattr(layer, "depthwise_kernel") or hasattr(layer, "pointwise_kernel"):
        return True
    else:
        return False


def contains_bias(layer):
    """
    Check whether the layer contains a bias.
    """

    # todo: add test and check this more throughroughly.
    # rely on Keras convention.
    if hasattr(layer, "bias"):
        return True
    else:
        return False


def only_relu_activation(layer):
    """Checks if layer contains no or only a ReLU activation."""
    return (not contains_activation(layer) or
            contains_activation(layer, None) or
            contains_activation(layer, "linear") or
            contains_activation(layer, "relu"))


def is_network(layer):
    """
    Is network in network?
    """
    if isinstance(layer, keras.layers.Layer):
        return False
    else:
        return True


def is_conv_layer(layer, *args, **kwargs):
    """Checks if layer is a convolutional layer."""
    CONV_LAYERS = (
        keras.layers.Conv1D,
        keras.layers.Conv2D,
        keras.layers.Conv2DTranspose,
        keras.layers.Conv3D,
        keras.layers.Conv3DTranspose,
        keras.layers.SeparableConv1D,
        keras.layers.SeparableConv2D,
        keras.layers.DepthwiseConv2D
    )
    return isinstance(layer, CONV_LAYERS)

def is_embedding_layer(layer, *_args, **_kwargs):
    """Checks if layer is an embedding layer."""
    return isinstance(layer, keras.layers.Embedding)

def is_batch_normalization_layer(layer, *args, **kwargs):
    """Checks if layer is a batchnorm layer."""
    return isinstance(layer, keras.layers.BatchNormalization)


def is_add_layer(layer, *args, **kwargs):
    """Checks if layer is an addition-merge layer."""
    return isinstance(layer, keras.layers.Add)


def is_dense_layer(layer, *args, **kwargs):
    """Checks if layer is a dense layer."""
    return isinstance(layer, keras.layers.core.Dense)


def is_convnet_layer(layer):
    """Checks if layer is from a convolutional network."""
    # Inside function to not break import if Keras changes.
    CONVNET_LAYERS = (
        keras.layers.InputLayer,
        keras.layers.ELU,
        keras.layers.LeakyReLU,
        keras.layers.PReLU,
        keras.layers.Softmax,
        keras.layers.ThresholdedReLU,
        keras.layers.Conv1D,
        keras.layers.Conv2D,
        keras.layers.Conv2DTranspose,
        keras.layers.Conv3D,
        keras.layers.Conv3DTranspose,
        keras.layers.Cropping1D,
        keras.layers.Cropping2D,
        keras.layers.Cropping3D,
        keras.layers.SeparableConv1D,
        keras.layers.SeparableConv2D,
        keras.layers.UpSampling1D,
        keras.layers.UpSampling2D,
        keras.layers.UpSampling3D,
        keras.layers.ZeroPadding1D,
        keras.layers.ZeroPadding2D,
        keras.layers.ZeroPadding3D,
        keras.layers.Activation,
        keras.layers.ActivityRegularization,
        keras.layers.Dense,
        keras.layers.Dropout,
        keras.layers.Flatten,
        keras.layers.Lambda,
        keras.layers.Masking,
        keras.layers.Permute,
        keras.layers.RepeatVector,
        keras.layers.Reshape,
        keras.layers.SpatialDropout1D,
        keras.layers.SpatialDropout2D,
        keras.layers.SpatialDropout3D,
        keras.layers.Embedding,
        keras.layers.LocallyConnected1D,
        keras.layers.LocallyConnected2D,
        keras.layers.Add,
        keras.layers.Average,
        keras.layers.Concatenate,
        keras.layers.Dot,
        keras.layers.Maximum,
        keras.layers.Minimum,
        keras.layers.Multiply,
        keras.layers.Subtract,
        keras.layers.AlphaDropout,
        keras.layers.GaussianDropout,
        keras.layers.GaussianNoise,
        keras.layers.BatchNormalization,
        keras.layers.AveragePooling1D,
        keras.layers.AveragePooling2D,
        keras.layers.AveragePooling3D,
        keras.layers.GlobalAveragePooling1D,
        keras.layers.GlobalAveragePooling2D,
        keras.layers.GlobalAveragePooling3D,
        keras.layers.GlobalMaxPooling1D,
        keras.layers.GlobalMaxPooling2D,
        keras.layers.GlobalMaxPooling3D,
        keras.layers.MaxPooling1D,
        keras.layers.MaxPooling2D,
        keras.layers.MaxPooling3D,
    )
    return isinstance(layer, CONVNET_LAYERS)


def is_relu_convnet_layer(layer):
    """Checks if layer is from a convolutional network with ReLUs."""
    return (is_convnet_layer(layer) and only_relu_activation(layer))


def is_average_pooling(layer):
    """Checks if layer is an average-pooling layer."""
    AVERAGEPOOLING_LAYERS = (
        keras.layers.AveragePooling1D,
        keras.layers.AveragePooling2D,
        keras.layers.AveragePooling3D,
        keras.layers.GlobalAveragePooling1D,
        keras.layers.GlobalAveragePooling2D,
        keras.layers.GlobalAveragePooling3D,
    )
    return isinstance(layer, AVERAGEPOOLING_LAYERS)


def is_max_pooling(layer):
    """Checks if layer is a max-pooling layer."""
    MAXPOOLING_LAYERS = (
        keras.layers.MaxPooling1D,
        keras.layers.MaxPooling2D,
        keras.layers.MaxPooling3D,
        keras.layers.GlobalMaxPooling1D,
        keras.layers.GlobalMaxPooling2D,
        keras.layers.GlobalMaxPooling3D,
    )
    return isinstance(layer, MAXPOOLING_LAYERS)


def is_input_layer(layer, ignore_reshape_layers=True):
    """Checks if layer is an input layer."""
    # Triggers if ALL inputs of layer are connected
    # to a Keras input layer object.
    # Note: In the sequential api the Sequential object
    # adds the Input layer if the user does not.
    kgraph = get_kgraph()

    layer_inputs = kgraph.get_input_layers(layer)
    # We ignore certain layers, that do not modify
    # the data content.
    # todo: update this list!
    IGNORED_LAYERS = (
        keras.layers.Flatten,
        keras.layers.Permute,
        keras.layers.Reshape,
    )
    while any([isinstance(x, IGNORED_LAYERS) for x in layer_inputs]):
        tmp = set()
        for l in layer_inputs:
            if(ignore_reshape_layers and
               isinstance(l, IGNORED_LAYERS)):
                tmp.update(kgraph.get_input_layers(l))
            else:
                tmp.add(l)
        layer_inputs = tmp

    if all([isinstance(x, keras.layers.InputLayer)
            for x in layer_inputs]):
        return True
    else:
        return False
