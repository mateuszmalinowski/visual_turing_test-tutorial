"""
Additional theano/keras functions.

Author: Mateusz Malinowski
Email: mmalinow@mpi-inf.mpg.de
"""

#import marshal
import numpy 
#import types

from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D

from keras.layers.core import Lambda
from keras.layers.core import MaskedLayer
from keras.layers.core import TimeDistributedMerge

from keras import backend as K


## functions ##
def time_distributed_nonzero_max_pooling(x):
    """
    Computes maximum along the first (time) dimension.
    It ignores the mask m.

    In:
        x - input; a 3D tensor
        mask_value - value to mask out, if None then no masking; 
            by default 0.0, 
    """

    import theano.tensor as T

    mask_value=0.0
    x = T.switch(T.eq(x, mask_value), -numpy.inf, x)
    masked_max_x = x.max(axis=1)
    # replace infinities with mask_value
    masked_max_x = T.switch(T.eq(masked_max_x, -numpy.inf), 0, masked_max_x)
    return masked_max_x


def time_distributed_masked_ave(x, m):
    """
    Computes average along the first (time) dimension.
    
    In:
        x - input; a 3D tensor
        m - mask
    """
    tmp = K.sum(x, axis=1)
    nonzeros = K.sum(m, axis=-1)
    return tmp / K.expand_dims(K.cast(nonzeros, tmp.dtype))


def time_distributed_masked_max(x, m):
    """
    Computes max along the first (time) dimension.

    In:
        x - input; a 3D tensor
        m - mask
        m_value - value for masking
    """
    # place infinities where mask is off
    m_value = 0.0
    tmp = K.switch(K.equal(m, 0.0), -numpy.inf, 0.0)
    x_with_inf = x + K.expand_dims(tmp)
    x_max = K.max(x_with_inf, axis=1) 
    r = K.switch(K.equal(x_max, -numpy.inf), m_value, x_max)
    return r 


## classes  ##

# Transforms existing layers to masked layers
class MaskedTimeDistributedMerge(MaskedLayer, TimeDistributedMerge):
    pass


class MaskedConvolution1D(MaskedLayer, Convolution1D):
    pass


class MaskedMaxPooling1D(MaskedLayer, MaxPooling1D):
    pass


# auxiliary mask-aware layers
class DropMask(MaskedLayer):
    """
    Removes a mask from the layer.
    """
    def get_output_mask(self, train=False):
        return None 


class LambdaWithMask(MaskedLayer, Lambda):
    """
    Lambda function that takes a two argument function, and returns
    a value returned by the function applied to the output of the previous layer
    and the mask.

    That is: LambdaWithMask(f) = f(previous, mask)
    """
    def get_output(self, train=False):
        #func = marshal.loads(self.function)
        #func = types.FunctionType(func, globals())
        func = self.function
        if hasattr(self, 'previous'):
            return func(self.previous.get_output(train), 
                    self.previous.get_output_mask(train))
        else:
            return func(self.input, self.get_output_mask(train))

