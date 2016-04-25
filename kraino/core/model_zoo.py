"""
Different Visual Turing Test architectures.

Author: Mateusz Malinowski
Email: mmalinow@mpi-inf.mpg.de
"""
import numpy as np

from theano import tensor as T

from keras.models import Sequential
from keras.models import Graph

from keras.layers.advanced_activations import ELU
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.advanced_activations import PReLU

from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D

from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Masking
from keras.layers.core import RepeatVector
from keras.layers.core import Reshape
#from keras.layers.core import Masking
from keras.layers.core import Merge
from keras.layers.core import TimeDistributedDense

from keras.layers.embeddings import Embedding

#from keras.layers.normalization import BatchNormalization

from keras_extensions import DropMask
from keras_extensions import LambdaWithMask
from keras_extensions import MaskedConvolution1D
from keras_extensions import MaskedTimeDistributedMerge
from keras_extensions import time_distributed_masked_ave
from keras_extensions import time_distributed_masked_max

from visual_model_zoo import select_sequential_visual_model

from ..utils.input_output_space import EOA


###
# Functions
###
def sample(scores, temperature=1.0):
    """
    Sampling words (each sample is drawn from a categorical distribution).

    In:
        scores - array of size #samples x #classes; 
            every entry determines a score for sample i having class j
        temperature - temperature for the predictions;
            the higher the flatter probabilities and hence more random answers

    Out:
        set of indices chosen as output, a vector of size #samples
    """
    logscores = np.log(scores) / temperature

    # numerically stable version
    normalized_logscores= logscores - np.max(logscores, axis=-1)[:, np.newaxis]
    margin_logscores = np.sum(np.exp(normalized_logscores),axis=-1)
    probs = np.exp(normalized_logscores) / margin_logscores[:,np.newaxis]

    draws = np.zeros_like(probs)
    num_samples = probs.shape[0]
    # we use 1 trial to mimic categorical distributions using multinomial
    for k in xrange(num_samples):
        draws[k,:] = np.random.multinomial(1,probs[k,:],1)
    return np.argmax(draws, axis=-1)


def max_likelihood(scores, **kwargs):
    """
    Pick up words that maximize the likelihood.
    """
    return np.argmax(scores, axis=-1)


###
# Extensible configuration class that encapsulates arguments for factories.
###
class Config(object):
    def __init__(self, 
            input_dim=None, 
            textual_embedding_dim=None, 
            visual_embedding_dim=None,
            hidden_state_dim=None,
            language_cnn_filter_size=None,
            language_cnn_filter_length=None,
            language_cnn_views=None,
            language_max_pool_length=None,
            output_dim=None, 
            visual_dim=None, 
            mlp_hidden_dim=None,
            merge_mode=None,
            multimodal_merge_mode=None,
            num_mlp_layers=None, 
            num_language_layers=None, 
            mlp_activation=None, 
            language_cnn_activation=None,
            fusion_layer_index=None,
            is_go_backwards=None,
            recurrent_encoder=None, 
            recurrent_decoder=None, 
            trainable_perception_name=None,
            word_generator=None,
            max_input_time_steps=None, 
            max_output_time_steps=None, 
            output_word_delimiter=None):
        """
        Every argument can be None if a given factory doesn't use it.
        The argument's list of this class is a union of arguments 
        of all factories.

        In:
            input_dim - dimension of the input space
            textual_embedding_dim - dimension of the textual embedding space
            visual_embedding_dim - dimension of the visual embedding space
            hidden_state_dim - dimension of the hidden space
            language_cnn_filter_size - number of language CNN filters
            language_cnn_filter_length - receptive field of language CNN filters
            language_cnn_views - number of language CNN views
            language_max_pool_length - receptive field of max pooling operator
            output_dim - dimension of the output space
            visual_dim - visual dimension;
                number of features for fixed, pre-trained models
                image dimensions for trainable models
            mlp_hidden_dim - hidden dimensionality for MLP
            merge_mode - general merge mode; how sequence of features are merged in temporal fusion
                \in \{'sum', 'mul', 'concat', 'ave'\}
            multimodal_merge_mode - defines how two or more modalities are merged
                \in \{'sum', 'mul', 'concat', 'ave'\}
            num_mlp_layers - number of layers for the MLP classifier
            num_language_layers - number of LSTM layers for language modeling
            mlp_activation - MLP activation function
            language_cnn_activation - language CNN activation function
            fusion_layer_index - specifies at which layer the fusion 
                of both models happens
            is_go_backwards - recurrent encoder processes input backwards if set True 
            recurrent_encoder - recurrent encoder
            recurrent_decoder - recurrent_decoder
            trainable_perception_name - name of the trainable perception;
                if this is 'none' then a pre-trained, fixed perception is used;
                only valid in multimodal models
            word_generatore - procedure to generate single words
            max_input_time_steps - maximal number of time steps in the input
            max_output_time_steps - maximal number of time steps in the output 
            output_word_delimiter - token that separates output words
        """
        self.input_dim = input_dim
        self.textual_embedding_dim = textual_embedding_dim
        self.visual_embedding_dim = visual_embedding_dim
        self.hidden_state_dim = hidden_state_dim
        self.language_cnn_filters = language_cnn_filter_size
        self.language_cnn_filter_length = language_cnn_filter_length
        self.language_cnn_views = language_cnn_views
        self.language_max_pool_length = language_max_pool_length
        self.output_dim = output_dim
        if type(visual_dim) is tuple and len(visual_dim) == 1:
            # we need to strip off the tuple
            self.visual_dim = visual_dim[0]
        else:
            self.visual_dim = visual_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.merge_mode = merge_mode
        self.multimodal_merge_mode = multimodal_merge_mode
        self.num_mlp_layers = num_mlp_layers
        self.num_language_layers = num_language_layers
        self.mlp_activation = mlp_activation
        self.language_cnn_activation = language_cnn_activation
        self.fusion_layer_index = fusion_layer_index
        self.go_backwards = is_go_backwards
        self.recurrent_encoder = recurrent_encoder
        self.recurrent_decoder = recurrent_decoder
        self.trainable_perception_name = trainable_perception_name
        self.word_generator = word_generator
        self.max_input_time_steps = max_input_time_steps
        self.max_output_time_steps = max_output_time_steps
        self.output_word_delimiter = output_word_delimiter
#


###
# Abstract classes
###
class AbstractModel(object):
    """
    Abstract Factory for different Visual Turing Test models.
    """
    def __init__(self, config):
        """
        If an argument is unusued in a concrete factory then use None as the 
        concrete argument.

        In:
            config - an object of the Config class
        """
        super(AbstractModel, self).__init__()
        self._config = config

    def create(self):
        """
        Creates the model.

        Out:
            built model
        """
        raise NotImplementedError()

    def get_type(self):
        """
        Returns a type of the network (Sequential or Graph).
        """
        raise NotImplementedError()


class CommonModel(AbstractModel): 
    """
    Represents common methods.
    """
    def deep_mlp(self):
        """
        Deep Multilayer Perceptrop.
        """
        if self._config.num_mlp_layers == 0:
            self.add(Dropout(0.5))
        else:
            for j in xrange(self._config.num_mlp_layers):
                self.add(Dense(self._config.mlp_hidden_dim))
                if self._config.mlp_activation == 'elu':
                    self.add(ELU())
                elif self._config.mlp_activation == 'leaky_relu':
                    self.add(LeakyReLU())
                elif self._config.mlp_activation == 'prelu':
                    self.add(PReLU())
                else:
                    self.add(Activation(self._config.mlp_activation))
                self.add(Dropout(0.5))

    def stacked_RNN(self, language_model, num_layers=None):
        """
        Stacked Recurrent Neural Network.
        """
        actual_num_layers = self._config.num_language_layers if num_layers == None \
                else num_layers
        if actual_num_layers <= 1:
            return language_model
        else:
            for k in xrange(actual_num_layers - 1):
                language_model.add(self._config.recurrent_encoder(
                    self._config.hidden_state_dim,
                    return_sequences=True, 
                    go_backwards=self._config.go_backwards))

    def textual_embedding(self, language_model, mask_zero):
        """
        Note:
        * mask_zero only makes sense if embedding is learnt
        """
        if self._config.textual_embedding_dim > 0:
            print('Textual Embedding is on')
            language_model.add(Embedding(
                self._config.input_dim, 
                self._config.textual_embedding_dim, 
                mask_zero=mask_zero))
        else:
            print('Textual Embedding is off')
            language_model.add(Reshape(
                input_shape=(self._config.max_input_time_steps, self._config.input_dim),
                dims=(self._config.max_input_time_steps, self._config.input_dim)))
            if mask_zero:
                language_model.add(Masking(0))
        return language_model

    def textual_embedding_fixed_length(self, language_model, mask_zero):
        """
        In contrast to textual_embedding, it produces a fixed length output.
        """
        if self._config.textual_embedding_dim > 0:
            print('Textual Embedding with fixed length is on')
            language_model.add(Embedding(
                self._config.input_dim, 
                self._config.textual_embedding_dim,
                input_length=self._config.max_input_time_steps,
                mask_zero=mask_zero))
        else:
            print('Textual Embedding with fixed length is off')
            language_model.add(Reshape(
                input_shape=(self._config.max_input_time_steps, self._config.input_dim),
                dims=(self._config.max_input_time_steps, self._config.input_dim)))
            if mask_zero:
                language_model.add(Masking(0))
        return language_model

    def visual_embedding(self, visual_model, input_dimensionality):
        if self._config.visual_embedding_dim > 0:
            print('Visual Embedding is on')
            visual_model.add(Dense(
                self._config.visual_embedding_dim, 
                input_shape=(input_dimensionality,)))
        return visual_model

    def temporal_pooling(self, model):
        #if self._config.merge_mode in ['sum', 'ave', 'mul']:
        if self._config.merge_mode in ['sum', 'mul']:
            model.add(MaskedTimeDistributedMerge(mode=self._config.merge_mode))
        elif self._config.merge_mode in ['ave']:
            model.add(LambdaWithMask(time_distributed_masked_ave, output_shape=[model.output_shape[2]]))
            #model.add(MaskedTimeDistributedMerge(mode='ave'))
        elif self._config.merge_mode in ['max']:
            model.add(LambdaWithMask(
                lambda x,m: time_distributed_masked_max(x,m), output_shape=[model.output_shape[2]]))
                #lambda x: x.max(axis=1), output_shape=[model.output_shape[2]]))
        else:
            raise NotImplementedError()
        return model


class AbstractSequentialModel(CommonModel, Sequential):
    def __init__(self, config):
        AbstractModel.__init__(self, config)
        Sequential.__init__(self)

    def get_type(self):
        return 'Sequential'


class AbstractGraphModel(CommonModel, Graph):
    def __init__(self, config):
        AbstractModel.__init__(self, config)
        Graph.__init__(self) 

    def get_type(self):
        return 'Graph'

    def compile(self, optimizer, loss, class_mode='categorical', theano_mode=None): 
        Graph.compile(self, optimizer, {self._output_name:loss})

    def fit(self, X, y, batch_size=128, nb_epoch=100, verbose=1, callbacks=[],
            validation_split=0., validation_data=None, shuffle=True, show_accuracy=False,
            class_weight=None, sample_weight=None):
        Graph.fit(self, {self._input_name:X, self._output_name:y}, 
                batch_size=batch_size,
                nb_epoch=nb_epoch,
                validation_split=validation_split,
                validation_data=validation_data,
                shuffle=shuffle,
                callbacks=callbacks)
                #class_weight=class_weight,
                #sample_weight=sample_weight)

    def predict(self, X, batch_size=128, verbose=0):
        return Graph.predict(self, {self._input_name:X}, 
                batch_size=batch_size,
                verbose=verbose)[self._output_name]


class AbstractSingleAnswer(object):
    def decode_predictions(self, X, temperature, index2word, 
            batch_size=512, verbose=0):
        """
        Decodes predictions 
        
        In:
            X - test set
            temperature - temperature for sampling
            index2word - mapping from word indices into word characters
            verbose - verbosity level, by default 0

        Out:
            Answer predictions (list of answers)
        """
        # preds is a matrix of size #samples x #words
        preds = self.predict(X, batch_size=batch_size,  verbose=verbose)
        answer_pred = map(
                lambda x:index2word[x], 
                self._config.word_generator(
                    scores=preds, temperature=temperature))
        return answer_pred


class AbstractSequentialMultiplewordAnswer(object):
    def decode_predictions(self, X, temperature, index2word, 
            batch_size=512, verbose=0):
        """
        Decodes predictions 
        
        In:
            X - test set
            temperature - temperature for sampling
            index2word - mapping from word indices into word characters
            verbose - verbosity level, by default 0

        Out:
            Answer predictions (list of answers)
        """
        # preds is a matrix of size #samples x #words
        if verbose > 0:
            print('Start predictions ...')
        preds = self.predict(X, batch_size=batch_size,  verbose=verbose)
        if verbose > 0:
            print('Decoding prediction ...')
        flattened_preds = preds.reshape(-1, preds.shape[-1])
        flattened_answer_pred = map(
                lambda x:index2word[x], 
                self._config.word_generator(
                    scores=flattened_preds, temperature=temperature))
        answer_pred_matrix = np.asarray(
                flattened_answer_pred).reshape(preds.shape[:2])
        answer_pred = []
        for a_no in answer_pred_matrix:
            a_no_modified = np.append(a_no, EOA)
            end_token_pos = [j for j,x in enumerate(a_no_modified) if x==EOA][0]
            tmp = self._config.output_word_delimiter.join(
                    a_no_modified[:end_token_pos])
            answer_pred.append(tmp)

        return answer_pred


###
# Blind models
###
class SequentialBlindTemporalFusionSingleAnswer(AbstractSequentialModel, AbstractSingleAnswer):
    """
    Sequential-based model that is:
    * blind
    * blank encoder (temporal fusion for encoding) 
    * decoder provides single word answers
    """
    def create(self):
        self.textual_embedding(self, mask_zero=True)
        self.temporal_pooling(self)
        self.add(DropMask())
        self.deep_mlp()
        self.add(Dense(self._config.output_dim))
        self.add(Activation('softmax'))


class SequentialBlindRecurrentFusionSingleAnswer(AbstractSequentialModel, AbstractSingleAnswer):
    """
    Sequential-based model that is: 
    * blind
    * decoder provides single word answers
    """
    def create(self):
        self.textual_embedding(self, mask_zero=True)
        self.stacked_RNN(self)
        self.add(self._config.recurrent_encoder(
            self._config.hidden_state_dim, 
            return_sequences=False,
            go_backwards=self._config.go_backwards))
        self.deep_mlp()
        self.add(Dense(self._config.output_dim))
        self.add(Activation('softmax'))


class SequentialBlindBidirectionalFusionDenseWordRepresentationSingleAnswer(
        AbstractSequentialModel, AbstractSingleAnswer):
    """
    Sequential-based model that is: 
    * blind
    * encoder is bidirectional
    * encoder on top of a dense word representation 
    * decoder provides single word answers

    Input: [textual, textual]
    """
    def create(self):

        assert self._config.textual_embedding_dim == 0, \
                'Embedding cannot be learnt but must be fixed'

        language_forward = Sequential()
        language_forward.add(self._config.recurrent_encoder(
            self._config.hidden_state_dim, return_sequences=False,
            input_shape=(self._config.max_input_time_steps, self._config.input_dim)))
        self.language_forward = language_forward

        language_backward = Sequential()
        language_backward.add(self._config.recurrent_encoder(
            self._config.hidden_state_dim, return_sequences=False,
            go_backwards=True,
            input_shape=(self._config.max_input_time_steps, self._config.input_dim)))
        self.language_backward = language_backward

        self.add(Merge([language_forward, language_backward]))
        self.deep_mlp()
        self.add(Dense(self._config.output_dim))
        self.add(Activation('softmax'))


class SequentialBlindRecurrenFusionSingleAnswerWithTemporalFusion(
        AbstractSequentialModel, AbstractSingleAnswer):
    """
    Sequential-based model that is:
    * blind
    * decoder provides single word answers by fusing all time steps
    """
    def create(self):
        assert self._config.merge_mode in ['max', 'ave', 'sum'], \
                'Merge mode of this model is either max, ave or sum'

        self.textual_embedding(self, mask_zero=False)
        self.stacked_RNN(self)
        self.add(self._config.recurrent_encoder(
            self._config.hidden_state_dim, 
            return_sequences=True,
            go_backwards=self._config.go_backwards))
        self.add(Dropout(0.5))
        self.add(TimeDistributedDense(self._config.output_dim))
        self.temporal_pooling(self)
        self.add(Activation('softmax'))


class SequentialBlindCNNFusionSingleAnswer(
        AbstractSequentialModel, AbstractSingleAnswer):
    """
    Sequential-based model that is:
    * blind
    * CNN encoder
    * decoder provides single word answers
    """
    def create(self):
        self.textual_embedding_fixed_length(self, mask_zero=False)
        self.add(Convolution1D(
            nb_filter=self._config.language_cnn_filters,
            filter_length=self._config.language_cnn_filter_length,
            border_mode='valid',
            activation=self._config.language_cnn_activation,
            subsample_length=1))
        self.add(MaxPooling1D(pool_length=self._config.language_max_pool_length))
        self.add(Flatten())
        self.deep_mlp()
        self.add(Dense(self._config.output_dim))
        self.add(Activation('softmax'))


class SequentialBlindCNNFusionSingleAnswerWithRecurrentFusion(
        AbstractSequentialModel, AbstractSingleAnswer):
    """
    Sequential-based model that is:
    * blind
    * CNN encoder
    * decoder provides single word answers with sequential fusion
    """
    def create(self):
        self.textual_embedding(self, mask_zero=False)
        self.add(Convolution1D(
            nb_filter=self._config.language_cnn_filters,
            filter_length=self._config.language_cnn_filter_length,
            border_mode='valid',
            activation=self._config.language_cnn_activation,
            subsample_length=1))
        #self.add(MaxPooling1D(pool_length=self._config.language_max_pool_length))
        self.add(self._config.recurrent_encoder(
            self._config.hidden_state_dim, 
            return_sequences=False,
            go_backwards=False))
        self.deep_mlp()
        self.add(Dense(self._config.output_dim))
        self.add(Activation('softmax'))


class SequentialBlindCNNFusionSingleAnswerWithTemporalFusion(
        AbstractSequentialModel, AbstractSingleAnswer):
    """
    Sequential-based model that is:
    * blind
    * CNN encoder
    * decoder provides single word answers by fusing all 'downsampled' time
      steps
    """
    def create(self):
        assert self._config.merge_mode in ['max', 'ave', 'sum'], \
                'Merge mode of this model is either max, ave or sum'

        self.textual_embedding(self, mask_zero=False)
        #self.textual_embedding(self, mask_zero=True)
        self.add(MaskedConvolution1D(
            nb_filter=self._config.language_cnn_filters,
            filter_length=self._config.language_cnn_filter_length,
            border_mode='valid',
            activation=self._config.language_cnn_activation,
            subsample_length=1))
        self.temporal_pooling(self)
        #self.add(DropMask())
        self.deep_mlp()
        self.add(Dense(self._config.output_dim))
        self.add(Activation('softmax'))


class SequentialBlind3ViewsCNNFusionSingleAnswerWithTemporalFusion(
        AbstractSequentialModel, AbstractSingleAnswer):
    """
    Sequential-based model that is:
    * blind
    * CNN encoder with 3 different filters (uni-, bi-, trigram)
    * decoder provides single word answers by fusing all 'downsampled' time
      steps

    Input: [textual, textual, textual]

    Note:
    If the embedding is learnt then the model doesn't share the embedding 
    parameters.

    Similar model to:
    * Y.Kim 'Convolutional neural networks for sentence classification'
    * Z.Yang et.al. 'Stacked Attention Networks for Image Question Answering'
    """
    def create(self):
        assert self._config.merge_mode in ['max', 'ave', 'sum'], \
                'Merge mode of this model is either max, ave or sum'

        unigram = Sequential() 
        self.textual_embedding(unigram, mask_zero=True)
        unigram.add(Convolution1D(
            nb_filter=self._config.language_cnn_filters,
            filter_length=1,
            border_mode='valid',
            activation=self._config.language_cnn_activation,
            subsample_length=1))
        self.temporal_pooling(unigram)

        bigram = Sequential()
        self.textual_embedding(bigram, mask_zero=True)
        bigram.add(Convolution1D(
            nb_filter=self._config.language_cnn_filters,
            filter_length=2,
            border_mode='valid',
            activation=self._config.language_cnn_activation,
            subsample_length=1))
        self.temporal_pooling(bigram)

        trigram = Sequential()
        self.textual_embedding(trigram, mask_zero=True)
        trigram.add(Convolution1D(
            nb_filter=self._config.language_cnn_filters,
            filter_length=3,
            border_mode='valid',
            activation=self._config.language_cnn_activation,
            subsample_length=1))
        self.temporal_pooling(trigram)

        self.add(Merge([unigram, bigram, trigram], mode='concat'))
        self.deep_mlp()
        self.add(Dense(self._config.output_dim))
        self.add(Activation('softmax'))


class SequentialBlindKViewsCNNFusionSingleAnswerWithTemporalFusion(
        AbstractSequentialModel, AbstractSingleAnswer):
    """
    Sequential-based model that is:
    * blind
    * CNN encoder with K different filters (uni-, bi-, trigram)
    * decoder provides single word answers by fusing all 'downsampled' time
      steps

    Input: [textual] * K Views

    Note:
    If the embedding is learnt then the model doesn't share the embedding 
    parameters.

    Generalization of the following models to K views:
    * Y.Kim 'Convolutional neural networks for sentence classification'
    * Z.Yang et.al. 'Stacked Attention Networks for Image Question Answering'
    """
    def create(self):
        assert self._config.merge_mode in ['max', 'ave', 'sum'], \
                'Merge mode of this model is either max, ave or sum'

        model_list = [None] * self._config.language_cnn_views
        for j in xrange(1,self._config.language_cnn_views+1):
            current_view = Sequential()
            self.textual_embedding(current_view, mask_zero=True)
            current_view.add(Convolution1D(
                nb_filter=self._config.language_cnn_filters,
                filter_length=j,
                border_mode='valid',
                activation=self._config.language_cnn_activation,
                subsample_length=1))
            self.temporal_pooling(current_view)
            model_list[j-1] = current_view

        self.add(Merge(model_list, mode='concat'))
        self.deep_mlp()
        self.add(Dense(self._config.output_dim))
        self.add(Activation('softmax'))


class SequentialBlindRecurrentFusionRecurrentMultiplewordAnswerDecoder(
        AbstractSequentialModel, AbstractSequentialMultiplewordAnswer):
    """
    Sequential-based model that is:
    * blind
    * decoder provides multiple word answers as a sequence
    """
    def create(self):
        self.textual_embedding(self, mask_zero=True)
        self.stacked_RNN(self)
        self.add(self._config.recurrent_encoder(
            self._config.hidden_state_dim, 
            return_sequences=False,
            go_backwards=self._config.go_backwards))
        self.add(Dropout(0.5))
        self.add(RepeatVector(self._config.max_output_time_steps))
        self.add(self._config.recurrent_decoder(
                self._config.hidden_state_dim, return_sequences=True))
        self.add(Dropout(0.5))
        self.add(TimeDistributedDense(self._config.output_dim))
        self.add(Activation('softmax'))


###
# Multimodal models
###
class SequentialMultimodalTemporalFusionSingleAnswer(
        AbstractSequentialModel, AbstractSingleAnswer):
    """
    Sequential-based model that is:
    * multimodal
    * blank encoder (temporal fusion)
    * decoder provides single word answers
    """
    def create(self):
        language_model = Sequential()
        self.textual_embedding(language_model, mask_zero=True)
        self.temporal_pooling(language_model)
        language_model.add(DropMask())
        #language_model.add(BatchNormalization(mode=1))
        self.language_model = language_model

        visual_model_factory = \
                select_sequential_visual_model[self._config.trainable_perception_name](
                    self._config.visual_dim)
        visual_model = visual_model_factory.create()
        visual_dimensionality = visual_model_factory.get_dimensionality()
        self.visual_embedding(visual_model, visual_dimensionality)
        #visual_model.add(BatchNormalization(mode=1))
        self.visual_model = visual_model
        
        if self._config.multimodal_merge_mode == 'dot':
            self.add(Merge([language_model, visual_model], mode='dot', dot_axes=[(1,),(1,)]))
        else:
            self.add(Merge([language_model, visual_model], mode=self._config.multimodal_merge_mode))

        self.deep_mlp()
        self.add(Dense(self._config.output_dim))
        self.add(Activation('softmax'))


class SequentialMultimodalRecurrentFusionAtLastTimestepMultimodalFusionSingleAnswer(
        AbstractSequentialModel, AbstractSingleAnswer):
    """
    Sequential-based model that is:
    * multimodal 
        * features are merged at last time step
    * decoder provides single word answers
    """
    def create(self):
        language_model = Sequential()
        self.textual_embedding(language_model, mask_zero=True)
        self.stacked_RNN(language_model)
        language_model.add(self._config.recurrent_encoder(
            self._config.hidden_state_dim, 
            return_sequences=False,
            go_backwards=self._config.go_backwards))
        self.language_model = language_model


        visual_model_factory = \
                select_sequential_visual_model[self._config.trainable_perception_name](
                    self._config.visual_dim)
        visual_model = visual_model_factory.create()
        visual_dimensionality = visual_model_factory.get_dimensionality()
        self.visual_embedding(visual_model, visual_dimensionality)
        self.visual_model = visual_model

        if self._config.multimodal_merge_mode == 'dot':
            self.add(Merge([language_model, visual_model], mode='dot', dot_axes=[(1,),(1,)]))
        else:
            self.add(Merge([language_model, visual_model], mode=self._config.multimodal_merge_mode))

        self.deep_mlp()
        self.add(Dense(self._config.output_dim))
        self.add(Activation('softmax'))


class SequentialMultimodalRecurrentFusionAtFirstTimestepMultimodalFusionSingleAnswer(
        AbstractSequentialModel, AbstractSingleAnswer):
    """
    Sequential-based model that is:
    * multimodal 
        * features are merged at first time step
    * decoder provides single word answers
    """
    def create(self):
        language_model = Sequential()
        self.textual_embedding(language_model, mask_zero=True)
        self.language_model = language_model

        visual_model_factory = \
                select_sequential_visual_model[self._config.trainable_perception_name](
                    self._config.visual_dim)
        visual_model = visual_model_factory.create()
        visual_dimensionality = visual_model_factory.get_dimensionality()
        self.visual_embedding(visual_model, visual_dimensionality)
        #visual_model = Sequential()
        #self.visual_embedding(visual_model)
        # the below should contain all zeros
        zero_model = Sequential()
        zero_model.add(RepeatVector(self._config.max_input_time_steps)-1)
        visual_model.add(Merge[visual_model, zero_model], mode='concat')
        self.visual_model = visual_model

        if self._config.multimodal_merge_mode == 'dot':
            self.add(Merge([language_model, visual_model], mode='dot', dot_axes=[(1,),(1,)]))
        else:
            self.add(Merge([language_model, visual_model], mode=self._config.multimodal_merge_mode))

        self.add(self._config.recurrent_encoder(
            self._config.hidden_state_dim, 
            return_sequences=False,
            go_backwards=self._config.go_backwards))
        self.deep_mlp()
        self.add(Dense(self._config.output_dim))
        self.add(Activation('softmax'))


class SequentialMultimodalRecurrentFusionAtEveryTimestepMultimodalFusionSingleAnswer(
        AbstractSequentialModel, AbstractSingleAnswer):
    """
    Sequential-based model that is:
    * multimodal 
        * features are merged at last time step with a dense adaptation layer
    * decoder provides single word answers
    """
    def create(self):
        language_model = Sequential()
        self.textual_embedding(language_model, mask_zero=True)
        self.language_model = language_model


        visual_model_factory = \
                select_sequential_visual_model[self._config.trainable_perception_name](
                    self._config.visual_dim)
        visual_model = visual_model_factory.create()
        visual_dimensionality = visual_model_factory.get_dimensionality()
        self.visual_embedding(visual_model, visual_dimensionality)
        #visual_model = Sequential()
        #self.visual_embedding(visual_model)
        self.visual_model = visual_model
        visual_model.add(RepeatVector(self._config.max_input_time_steps))

        if self._config.multimodal_merge_mode == 'dot':
            self.add(Merge([language_model, visual_model], mode='dot', dot_axes=[(1,),(1,)]))
        else:
            self.add(Merge([language_model, visual_model], mode=self._config.multimodal_merge_mode))

        self.add(self._config.recurrent_encoder(
            self._config.hidden_state_dim, 
            return_sequences=False,
            go_backwards=self._config.go_backwards))
        self.deep_mlp()
        self.add(Dense(self._config.output_dim))
        self.add(Activation('softmax'))


class SequentialMultimodalRecurrentFusionAtLastTimestepMultimodalFusionRecurrentMultiplewordAnswerDecoder(
        AbstractSequentialModel, AbstractSequentialMultiplewordAnswer):
    """
    Sequential-based model that is:
    * multimodal 
        * features are merged at last time step with a dense adaptation layer
    * decoder provides multiple word answers as a sequence
    """
    def create(self):
        language_model = Sequential()
        self.textual_embedding(language_model, mask_zero=True)
        self.stacked_RNN(language_model)
        language_model.add(self._config.recurrent_encoder(
            self._config.hidden_state_dim, 
            return_sequences=False,
            go_backwards=self._config.go_backwards))
        self.language_model = language_model

        visual_model_factory = \
                select_sequential_visual_model[self._config.trainable_perception_name](
                    self._config.visual_dim)
        visual_model = visual_model_factory.create()
        visual_dimensionality = visual_model_factory.get_dimensionality()
        self.visual_embedding(visual_model, visual_dimensionality)
        #visual_model = Sequential()
        #self.visual_embedding(visual_model)
        self.visual_model = visual_model

        if self._config.multimodal_merge_mode == 'dot':
            self.add(Merge([language_model, visual_model], mode='dot', dot_axes=[(1,),(1,)]))
        else:
            self.add(Merge([language_model, visual_model], mode=self._config.multimodal_merge_mode))

        self.add(Dropout(0.5))
        self.add(Dense(self._config.output_dim))

        self.add(RepeatVector(self._config.max_output_time_steps))
        self.add(self._config.recurrent_decoder(
                self._config.hidden_state_dim, return_sequences=True))
        self.add(Dropout(0.5))
        self.add(TimeDistributedDense(self._config.output_dim))
        self.add(Activation('softmax'))


###
# Graph-based models
###
class GraphBlindSingleAnswer(AbstractGraphModel, AbstractSingleAnswer):
    """
    Graph-based model that is:
    * blind
    * decoder provides single word answers

    Original work: Ashkan Mokarian [ashkan@mpi-inf.mpg.de]
    TODO: Currently not supported!
    """
    def create(self):
        self._input_name = 'text'
        self._output_name = 'output'

        self.add_input(
                name=self._input_name, 
                input_shape=(self._config.max_input_time_steps, self._config.input_dim,))
        self.inputs['text'].input = T.imatrix()
        self.add_node(Embedding(
                self._config.input_dim, 
                self._config.textual_embedding_dim, 
                mask_zero=True), 
                name='embedding', input='text')
        self.add_node(
                self._config.recurrent_encoder(
                    self._config.hidden_state_dim, 
                    return_sequences=False,
                    go_backwards=self._config.go_backwards),
                name='recurrent', input='embedding') 
        self.add_node(Dropout(0.5), name='dropout', input='recurrent')
        self.add_node(Dense(self._config.output_dim), name='dense', input='dropout')
        self.add_node(Activation('softmax'), name='softmax', input='dense')
        self.add_output(name=self._output_name, input='softmax')


class GraphBlindBidirectionalSingleAnswer(
        AbstractGraphModel, AbstractSingleAnswer):
    """
    Graph-based model that is: 
    * blind
    * encoder is bidirectional
    * decoder provides single word answers

    TODO:
    * Kraino doesn't have a good support for graph-based models
    """
    def create(self):
        self._input_name = 'text'
        self._output_name = 'output'

        self.add_input(
                name=self._input_name,
                input_shape=(self._config.max_input_time_steps, self._config.input_dim,))
        self.inputs['text'].input = T.imatrix()
        self.add_node(self._config.recurrent_encoder(
            self._config.hidden_state_dim, return_sequences=False), 
            name='forward', input='text')
        self.add_node(self._config.recurrent_encoder(
            self._config.hidden_state_dim, return_sequences=False),
            name='backward', input='text')
        self.add_node(Dropout(0.5), 
                name='pre_classification_dropout',
                inputs=['forward', 'backward'],
                merge_mode=self._config.merge_mode)
        self.add_node(Dense(self._config.output_dim), 
                name='classification_dense',
                input='pre_classification_dropout')
        self.add_node(Activation('softmax'),
                name='classification_softmax',
                input='classification_dense')


###
# Selects factory
###
select = \
        {
        'sequential-blind-temporal_fusion-single_answer': \
                SequentialBlindTemporalFusionSingleAnswer,
        'sequential-blind-recurrent_fusion-single_answer': \
                SequentialBlindRecurrentFusionSingleAnswer,
        'sequential-blind-bidirectional_fusion-dense_word_representation-single_answer': \
                SequentialBlindBidirectionalFusionDenseWordRepresentationSingleAnswer,
        'sequential-blind-recurrent_fusion-single_answer_with_temporal_fusion': \
                SequentialBlindRecurrenFusionSingleAnswerWithTemporalFusion,
        'sequential-blind-cnn_fusion-single_answer': \
                SequentialBlindCNNFusionSingleAnswer,
        'sequential-blind-cnn_fusion-single_answer_with_recurrent_fusion': \
                SequentialBlindCNNFusionSingleAnswerWithRecurrentFusion,
        'sequential-blind-cnn_fusion-single_answer_with_temporal_fusion': \
                SequentialBlindCNNFusionSingleAnswerWithTemporalFusion,
        'sequential-blind-cnn_3views_fusion-single_answer_with_temporal_fusion': \
                SequentialBlind3ViewsCNNFusionSingleAnswerWithTemporalFusion,
        'sequential-blind-cnn_kviews_fusion-single_answer_with_temporal_fusion': \
                SequentialBlindKViewsCNNFusionSingleAnswerWithTemporalFusion,
        'sequential-blind-recurrent_fusion-recurrent_multipleword_answer_decoder': \
                SequentialBlindRecurrentFusionRecurrentMultiplewordAnswerDecoder,
        'sequential-multimodal-temporal_fusion-single_answer': \
                SequentialMultimodalTemporalFusionSingleAnswer,
        'sequential-multimodal-recurrent_fusion-at_last_timestep_multimodal_fusion-single_answer': \
                SequentialMultimodalRecurrentFusionAtLastTimestepMultimodalFusionSingleAnswer,
        'sequential-multimodal-recurrent_fusion-at_first_timestep_multimodal_fusion-single_answer': \
                SequentialMultimodalRecurrentFusionAtFirstTimestepMultimodalFusionSingleAnswer,
        'sequential-multimodal-recurrent_fusion-at_every_timestep_multimodal_fusion-single_answer': \
                SequentialMultimodalRecurrentFusionAtEveryTimestepMultimodalFusionSingleAnswer,
        'sequential-multimodal-recurrent_fusion-at_last_timestep_multimodal_fusion-recurrent_multipleword_answer_decoder': \
                SequentialMultimodalRecurrentFusionAtLastTimestepMultimodalFusionRecurrentMultiplewordAnswerDecoder,
        'graph-blind-single_answer': GraphBlindSingleAnswer,
        'graph-blind-bidirectional-single_answer': GraphBlindBidirectionalSingleAnswer
        }

word_generator = \
        {
        'sample': sample,
        'max_likelihood': max_likelihood,
        }
