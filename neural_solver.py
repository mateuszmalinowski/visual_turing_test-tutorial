#!/usr/bin/env python
from __future__ import print_function

"""
QA model.
Uses embedding.

Implementation in Keras.

Author: Mateusz Malinowski
Email: mmalinow@mpi-inf.mpg.de
"""

import os
import numpy as np

from socket import gethostname
from spacy.en import English
from toolz import compose
from toolz import frequencies 
from timeit import default_timer as timer

from keras.optimizers import Adadelta
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import SGD

from keras.preprocessing import sequence

from kraino.core import recurrent_net 
from kraino.core import model_zoo
from kraino.core.model_zoo import Config
from kraino.core.visual_model_zoo import get_visual_features
from kraino.core.visual_model_zoo import imagenet_mean_preprocess_image_tensor_fun

from kraino.utils import data_provider
from kraino.utils.read_write import pickle_model
#from kraino.utils.read_write import model_to_json
from kraino.utils.parsers import parse_input_arguments
from kraino.utils.callbacks import StoreModelWeightsOnEraEnd
from kraino.utils.callbacks import PrintOnEraBegin 
from kraino.utils.callbacks import PrintPerformanceMetricOnEraEnd
from kraino.utils.callbacks import MonitorPredictionsOnEndEra
#from kraino.utils.callbacks import PlotPerformanceMetricOnEraEnd
#from kraino.utils.callbacks import StandardPerformancePlot
from kraino.utils.callbacks import LearningRateReducerWithEarlyStopping
from kraino.utils.callbacks import LearningRateReducerEveryPatienceEpoch
from kraino.utils.input_output_space import build_vocabulary
from kraino.utils.input_output_space import encode_questions_index
from kraino.utils.input_output_space import encode_questions_dense
from kraino.utils.input_output_space import encode_answers_one_hot
#from kraino.utils.model_visualization import model_picture

from theano import config as theano_config


###
# Constants
###
# path to the outputted verbosity
VERBOSITY_PATH_PREFIX = os.path.join('local', 'logs', 'verbosity')

# class normalized logs
CLASS_NORMALIZED_PATH_PREFIX = os.path.join('local', 'logs', 'class_normalized')

# path where the weights are saved
WEIGHTS_PATH_PREFIX = os.path.join('local', 'weights', 'weights')

OPTIMIZERS = { \
        'sgd':SGD,
        'adagrad':Adagrad,
        'adadelta':Adadelta,
        'rmsprop':RMSprop,
        'adam':Adam, 
        }
###

###
# Functions
###
def main(params):
    verbosity_tmp = params['VERBOSITY']

    # seperate verbosity levels by ':' or ',' or ' ' (space)
    if not verbosity_tmp == '':
        if ':' in verbosity_tmp :
            delimiter = ':'
        elif ',' in verbosity_tmp:
            delimiter = ','
        else:
            delimiter = ' '
        verbosity = verbosity_tmp.split(delimiter)

        verbosity_path_longprefix = VERBOSITY_PATH_PREFIX + \
                '.' + params['MODEL'] + '.' + params['DATASET'] + \
                '.' + gethostname() + '.' + theano_config.device + \
                '.epoch_{epoch:02d}.era_{era:02d}'
    
        class_normalized_path_longprefix = CLASS_NORMALIZED_PATH_PREFIX  + \
                '.' + params['MODEL'] + '.' + params['DATASET'] + \
                '.' + gethostname() + '.' + theano_config.device + \
                '.epoch_{epoch:02d}.era_{era:02d}'
    else:
        verbosity = []

    model_path=os.path.join('local', 'models', 'model.{0}.{1}.pkl'.format(
            params['MODEL'], params['DATASET']))

    era_weights_loader = params['WEIGHTS_LOADER_ERA']
    if era_weights_loader >= 0:
        is_load_weights = True
    else:
        is_load_weights = False

    if params['WEIGHTS_LOADER_NAME'] == '':
        weights_loader_name = gethostname()
    else:
        weights_loader_name = params['WEIGHTS_LOADER_NAME']
    weights_path = WEIGHTS_PATH_PREFIX + '.' + \
            params['MODEL'] + '.' + params['DATASET'] + \
            '.' + weights_loader_name + \
            '.epoch_{epoch:02d}.era_{era:02d}.hdf5'

    ###
    # Load the dataset
    ###
    dp = data_provider.select[params['DATASET']]
    train_dataset = dp['text'](
            train_or_test='train',
            answer_mode=params['VQA_ANSWER_MODE'],
            keep_top_qa_pairs=params['NUM_TOP_QA_PAIRS'])
    train_x, train_y = train_dataset['x'], train_dataset['y']
    print('Number of training examples {0}'.format(len(train_x)))
    train_image_names = train_dataset['img_name']
    train_visual_features = get_visual_features(
            data_provider=dp,
            trainable_perception_name=params['TRAINABLE_PERCEPTION_NAME'],
            train_or_test='train',
            image_names_list=train_image_names,
            parts_extractor=params['PARTS_EXTRACTOR'],
            max_parts=params['MAX_MEMORY_TIME_STEPS'],
            perception=params['PERCEPTION'],
            layer=params['PERCEPTION_LAYER'],
            second_layer=params['PERCEPTION_SECOND_LAYER'],
            preprocess_image_tensor_fun=imagenet_mean_preprocess_image_tensor_fun)
    train_question_id = train_dataset['question_id']
    if np.any([params['IS_VALIDATION_SET'], 
            'monitor_val_metric' in verbosity,
            'monitor_val_predictions' in verbosity,
            'plot_val_metric' in verbosity]):
        val_dataset = dp['text'](
                train_or_test='val',
                answer_mode='single_confident',
                keep_top_qa_pairs=0)
        val_x, val_y = val_dataset['x'], val_dataset['y']
        val_question_id = val_dataset['question_id']
        val_image_names = val_dataset['img_name']
        val_visual_features = get_visual_features(
                data_provider=dp,
                trainable_perception_name=params['TRAINABLE_PERCEPTION_NAME'],
                train_or_test='val',
                image_names_list=val_image_names,
                parts_extractor=params['PARTS_EXTRACTOR'],
                max_parts=params['MAX_MEMORY_TIME_STEPS'],
                perception=params['PERCEPTION'],
                layer=params['PERCEPTION_LAYER'],
                second_layer=params['PERCEPTION_SECOND_LAYER'],
                preprocess_image_tensor_fun=imagenet_mean_preprocess_image_tensor_fun)
    if np.any(['monitor_test_metric' in verbosity, 
            'monitor_test_predictions' in verbosity,
            'plot_test_metric' in verbosity]):
        test_dataset = dp['text'](
                train_or_test='test',
                answer_mode='single_confident',
                keep_top_qa_pairs=0)
        test_x, test_y = test_dataset['x'], test_dataset['y']
        test_question_id = test_dataset['question_id']
        test_image_names = test_dataset['img_name']
        test_visual_features = get_visual_features(
                data_provider=dp,
                trainable_perception_name=params['TRAINABLE_PERCEPTION_NAME'],
                train_or_test='test',
                image_names_list=test_image_names,
                parts_extractor=params['PARTS_EXTRACTOR'],
                max_parts=params['MAX_MEMORY_TIME_STEPS'],
                perception=params['PERCEPTION'],
                layer=params['PERCEPTION_LAYER'],
                second_layer=params['PERCEPTION_SECOND_LAYER'],
                preprocess_image_tensor_fun=imagenet_mean_preprocess_image_tensor_fun)

    ### 
    # Building vocabularies
    ###
    split_symbol = '{'
    if type(train_x[0]) is unicode:
        # choose a split symbol that doesn't exist in text
        split_function = lambda x: unicode.split(x, split_symbol)
    elif type(train_x[0]) is str:
        split_function = lambda x: str.split(x, split_symbol)
    else:
        raise NotImplementedError() 

    wordcount = compose(frequencies, split_function)
    wordcount_x = wordcount(split_symbol.join(train_x).replace(' ',split_symbol))
    if params['IS_WHOLE_ANSWER_AS_ANSWER_WORD']:
        wordcount_y = wordcount(split_symbol.join(train_y))
    else:
        wordcount_y = wordcount(split_symbol.join(train_y).replace(
            train_dataset['answer_words_delimiter'],split_symbol))

    word2index_x, index2word_x = build_vocabulary(
            this_wordcount=wordcount_x, 
            is_reset=True,
            truncate_to_most_frequent=params['TRUNCATE_INPUT_SPACE'])
    word2index_y, index2word_y = build_vocabulary(
            this_wordcount=wordcount_y, 
            is_reset=True,
            truncate_to_most_frequent=params['TRUNCATE_OUTPUT_SPACE'])

    print('Size of the input {0}, and output vocabularies {1}'.\
            format(len(word2index_x), len(word2index_y)))

    # save vocabulary
    ###

    ###
    # Building input/output
    # Dimensions: 
    #   data points
    #   time steps 
    #   encodings of the words
    ###
    if params['WORD_REPRESENTATION'] == 'one_hot':
        one_hot_x = encode_questions_index(train_x, word2index_x)
        X_train = sequence.pad_sequences(one_hot_x, maxlen=params['MAX_INPUT_TIME_STEPS'])
    elif params['WORD_REPRESENTATION'] == 'dense':
        word_encoder = English()
        X_train = encode_questions_dense(
                x=train_x, 
                word_encoder=word_encoder, 
                max_time_steps=params['MAX_INPUT_TIME_STEPS'],
                is_remove_question_symbol=True)
    else:
        raise NotImplementedError()
    if params['IS_WHOLE_ANSWER_AS_ANSWER_WORD']:
        train_answer_words_delimiter = None
    else:
        train_answer_words_delimiter = train_dataset['answer_words_delimiter']
    Y, train_y_gt = encode_answers_one_hot(train_y, word2index_y,
            max_answer_time_steps=params['MAX_OUTPUT_TIME_STEPS'],
            is_only_first_answer_word=params['IS_ONLY_FIRST_ANSWER_WORD'],
            answer_words_delimiter=train_answer_words_delimiter)

    if '-bidirectional-' in params['MODEL']:
        train_input = [X_train] * 2
    elif '-cnn_3views-' in params['MODEL']:
        train_input = [X_train] * 3
    elif '-cnn_kviews-' in params['MODEL']:
        train_input = [X_train] * params['LANGUAGE_CNN_VIEWS']
    else:
        train_input = [X_train]

    if '-multimodal-' in params['MODEL']:
        train_input.append(train_visual_features)

    if np.any([params['IS_VALIDATION_SET'], 
            'monitor_val_metric' in verbosity,
            'monitor_val_predictions' in verbosity,
            'plot_val_metric' in verbosity]):
        if params['WORD_REPRESENTATION'] == 'one_hot':
            one_hot_x_val= encode_questions_index(val_x, word2index_x)
            X_val = sequence.pad_sequences(one_hot_x_val, 
                    maxlen=params['MAX_INPUT_TIME_STEPS'])
        elif params['WORD_REPRESENTATION'] == 'dense':
            X_val = encode_questions_dense(
                x=val_x, 
                word_encoder=word_encoder, 
                max_time_steps=params['MAX_INPUT_TIME_STEPS'],
                is_remove_question_symbol=True)
        else:
            NotImplementedError()
 
        if params['IS_WHOLE_ANSWER_AS_ANSWER_WORD']:
            val_answer_words_delimiter = None
        else:
            val_answer_words_delimiter = val_dataset['answer_words_delimiter']
        Y_val, _ = encode_answers_one_hot(val_y, word2index_y, 
                max_answer_time_steps=params['MAX_OUTPUT_TIME_STEPS'],
                is_only_first_answer_word=params['IS_ONLY_FIRST_ANSWER_WORD'],
                answer_words_delimiter=val_answer_words_delimiter)
        if '-bidirectional-' in params['MODEL']:
            val_input = [X_val] * 2
        elif '-cnn_3views-' in params['MODEL']:
            val_input = [X_val] * 3
        elif '-cnn_kviews-' in params['MODEL']:
            val_input = [X_val] * params['LANGUAGE_CNN_VIEWS']
        else:
            val_input = [X_val]

        if '-multimodal-' in params['MODEL']:
            val_input.append(val_visual_features)
        validation_set = (val_input, Y_val)

    if np.any(['monitor_test_metric' in verbosity, 
            'monitor_test_predictions' in verbosity,
            'plot_test_metric' in verbosity]):
        if params['WORD_REPRESENTATION'] == 'one_hot':
            one_hot_x_test = encode_questions_index(test_x, word2index_x)
            X_test = sequence.pad_sequences(one_hot_x_test, 
                    maxlen=params['MAX_INPUT_TIME_STEPS'])
        elif params['WORD_REPRESENTATION'] == 'dense':
            X_test = encode_questions_dense(
                    x=test_x,
                    word_encoder=word_encoder,
                    max_time_steps=params['MAX_INPUT_TIME_STEPS'],
                    is_remove_question_symbol=True)
        else:
            NotImplementedError()
        if '-bidirectional-' in params['MODEL'] \
                and 'sequential-blind' in params['MODEL']:
            test_input = [X_test] * 2
        elif '-cnn_3views-' in params['MODEL']:
            test_input = [X_test] * 3
        elif '-cnn_kviews-' in params['MODEL']:
            test_input = [X_test] * params['LANGUAGE_CNN_VIEWS']
        else:
            test_input = [X_test]
        if '-multimodal-' in params['MODEL']:
            test_input.append(test_visual_features)

    # convert to numpy arrays
    # train_y - original training ys
    # train_y_gt - training ys used to learn the model
    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    train_y_gt = np.asarray(train_y_gt)
    ###

    ###
    # Callbacks
    ###
    callbacks = []
    if params['IS_SAVE_WEIGHTS']:
        callback_store_model = StoreModelWeightsOnEraEnd(
                filepath=weights_path,
                epoch_interval=params['MAX_EPOCH'])
        callbacks.append(callback_store_model)

    callback_print_on_era_begin = PrintOnEraBegin(
            epoch_interval=params['MAX_EPOCH'],
            message="Era {era:2d} out of " + str(params['MAX_ERA']))
    callbacks.append(callback_print_on_era_begin)

    # verbosity callbacks
    if 'monitor_val_metric' in verbosity:
        if params['METRIC'] == 'vqa':
            results_function = lambda x: \
                    val_dataset['vqa_object'].loadRes(
                            x, val_dataset['questions_path'])
            extra_vars = {
                    'question_id':val_question_id,
                    'vqa_object':val_dataset['vqa_object'],
                    'resfun':results_function}
        else:
            extra_vars = None
        callback_metric = PrintPerformanceMetricOnEraEnd(
                X=val_input, 
                y=val_y,
                temperature=params['TEMPERATURE'],
                index2word_y=index2word_y,
                metric_name=params['METRIC'],
                epoch_interval=params['MAX_EPOCH'],
                extra_vars=extra_vars,
                verbosity_path=class_normalized_path_longprefix+'.val.acc',
                verbose=1)
        callbacks.append(callback_metric)

    if 'monitor_test_metric' in verbosity:
        if params['METRIC'] == 'vqa':
            results_function = lambda x: \
                    test_dataset['vqa_object'].loadRes(
                            x, test_dataset['questions_path'])
            extra_vars = {
                    'question_id':test_question_id,
                    'vqa_object':test_dataset['vqa_object'],
                    'resfun':results_function}
        else:
            extra_vars = None
        callback_metric = PrintPerformanceMetricOnEraEnd(
                X=test_input, 
                y=test_y,
                temperature=params['TEMPERATURE'],
                index2word_y=index2word_y,
                metric_name=params['METRIC'],
                epoch_interval=params['MAX_EPOCH'],
                extra_vars=extra_vars,
                verbosity_path=class_normalized_path_longprefix+'.test.acc',
                verbose=1)
        callbacks.append(callback_metric)

    if 'monitor_train_predictions' in verbosity:
        callback_monitor_train_predictions = MonitorPredictionsOnEndEra(
            X=train_input, x=train_x, y=train_y,
            temperature=params['TEMPERATURE'],
            index2word_y=index2word_y,
            verbosity_path=verbosity_path_longprefix+'.train.pred',
            epoch_interval=params['MAX_EPOCH'],
            subset_size=params['TRAINING_SUBSET_SIZE'],
            verbose=1)
        callbacks.append(callback_monitor_train_predictions)

    if 'monitor_val_predictions' in verbosity:
        callback_monitor_train_predictions = MonitorPredictionsOnEndEra(
            X=val_input, x=val_x, y=val_y,
            temperature=params['TEMPERATURE'],
            index2word_y=index2word_y,
            verbosity_path=verbosity_path_longprefix+'.val.pred',
            epoch_interval=params['MAX_EPOCH'],
            subset_size=params['VAL_SUBSET_SIZE'],
            verbose=1)
        callbacks.append(callback_monitor_train_predictions)

    if 'monitor_test_predictions' in verbosity:
        callback_monitor_train_predictions = MonitorPredictionsOnEndEra(
            X=test_input, x=test_x, y=test_y,
            temperature=params['TEMPERATURE'],
            index2word_y=index2word_y,
            verbosity_path=verbosity_path_longprefix+'.test.pred',
            epoch_interval=params['MAX_EPOCH'],
            subset_size=params['TEST_SUBSET_SIZE'],
            verbose=1)
        callbacks.append(callback_monitor_train_predictions)

    """
    if 'plot_loss' in verbosity:
        callback_plot_loss = StandardPerformancePlot(
                name='Kraino', 
                fig_title=params['VISUALIZATION_FIG_LOSS_TITLE'], 
                url=params['VISUALIZATION_URL'])
        callbacks.append(callback_plot_loss)

    if 'plot_train_acc' in verbosity:
        callback_plot_acc = StandardPerformancePlot(
                name='Kraino',
                fig_title='Acc',
                url=params['VISUALIZATION_URL'],
                what_to_plot=['acc'])

    if 'plot_trainval_acc' in verbosity:
        callback_plot_acc = StandardPerformancePlot(
                name='Kraino', 
                fig_title='Acc', 
                url=params['VISUALIZATION_URL'],
                what_to_plot=['acc', 'val_acc'])
        callbacks.append(callback_plot_acc)

    if 'plot_train_metric' in verbosity:
        if params['METRIC'] == 'vqa':
            results_function = lambda x: \
                    train_dataset['vqa_object'].loadRes(
                            x, val_dataset['questions_path'])
            extra_vars = {
                    'question_id':train_question_id,
                    'vqa_object':train_dataset['vqa_object'],
                    'resfun':results_function,
                    }
        else:
            extra_vars=None
        callback_plot_train_metric = PlotPerformanceMetricOnEraEnd(
                X=train_input, 
                y=train_y,
                temperature=params['TEMPERATURE'],
                index2word_y=index2word_y,
                metric_name=params['METRIC'],
                epoch_interval=params['MAX_EPOCH'],
                extra_vars=extra_vars,
                verbose=1,
                name='Kraino',
                fig_title='{0} - {1}'.format(
                    params['VISUALIZATION_FIG_METRIC_TITLE'], 'train'),
                url=params['VISUALIZATION_URL'])
        callbacks.append(callback_plot_train_metric)

    if 'plot_val_metric' in verbosity:
        if params['METRIC'] == 'vqa':
            results_function = lambda x: \
                    val_dataset['vqa_object'].loadRes(
                            x, val_dataset['questions_path'])
            extra_vars = {
                    'question_id':val_question_id,
                    'vqa_object':val_dataset['vqa_object'],
                    'resfun':results_function,
                    }
        else:
            extra_vars=None
        callback_plot_val_metric = PlotPerformanceMetricOnEraEnd(
                X=val_input, 
                y=val_y,
                temperature=params['TEMPERATURE'],
                index2word_y=index2word_y,
                metric_name=params['METRIC'],
                epoch_interval=params['MAX_EPOCH'],
                extra_vars=extra_vars,
                verbose=1,
                name='Kraino',
                fig_title='{0} - {1}'.format(
                    params['VISUALIZATION_FIG_METRIC_TITLE'], 'val'),
                url=params['VISUALIZATION_URL'])
        callbacks.append(callback_plot_val_metric)

    if 'plot_test_metric' in verbosity:
        if params['METRIC'] == 'vqa':
            results_function = lambda x: \
                    test_dataset['vqa_object'].loadRes(
                            x, test_dataset['questions_path'])
            extra_vars = {
                    'question_id':test_question_id,
                    'vqa_object':test_dataset['vqa_object'],
                    'resfun':results_function}
        else:
            extra_vars = None
        callback_plot_test_metric = PlotPerformanceMetricOnEraEnd(
                X=test_input, 
                y=test_y,
                temperature=params['TEMPERATURE'],
                index2word_y=index2word_y,
                metric_name=params['METRIC'],
                epoch_interval=params['MAX_EPOCH'],
                extra_vars=extra_vars,
                verbose=1,
                name='Kraino',
                fig_title='{0} - {1}'.format(
                    params['VISUALIZATION_FIG_METRIC_TITLE'], 'test'),
                url=params['VISUALIZATION_URL'])
        callbacks.append(callback_plot_test_metric)
    """

    # training changers
    if params['IS_EARLY_STOPPING']: 
        callback_lr_reducer = LearningRateReducerWithEarlyStopping(
                patience=params['LR_PATIENCE'],
                reduce_rate=params['REDUCE_RATE'],
                reduce_nb=params['MAX_NUMBER_REDUCTIONS'],
                is_early_stopping=params['IS_EARLY_STOPPING'])
        callbacks.append(callback_lr_reducer)

    if params['IS_LR_FIXED_REDUCTION']:
        # reduction after fixed number of epochs
        callback_lr_reducer_after_k_epoch = LearningRateReducerEveryPatienceEpoch(
                patience=params['LR_PATIENCE'],
                reduce_rate=params['REDUCE_RATE'],
                reduce_nb=params['MAX_NUMBER_REDUCTIONS'])
        callbacks.append(callback_lr_reducer_after_k_epoch)



    print('Our callbacks: ' + str(callbacks))
    ###
 
    ###
    # Building model
    ###
    print('Building model ...')
    #nb_words= max(X_train.flatten())+1
    input_dim = len(word2index_x.keys()) \
            if params['WORD_REPRESENTATION'] == 'one_hot' \
            else X_train[0][0].shape[0]
    output_dim = len(word2index_y.keys())
    visual_dim = train_visual_features.shape[1:] \
            if train_visual_features is not None else 0
    # creating the config object that carries arguments for models
    model_config = Config( 
            input_dim=input_dim, 
            textual_embedding_dim=0 if params['WORD_REPRESENTATION'] == 'dense' 
                    else params['TEXTUAL_EMBEDDING_SIZE'], 
            visual_embedding_dim=params['VISUAL_EMBEDDING_SIZE'],
            hidden_state_dim=params['HIDDEN_STATE_SIZE'],
            language_cnn_filter_size=params['LANGUAGE_CNN_FILTERS'],
            language_cnn_filter_length=params['LANGUAGE_CNN_FILTER_LENGTH'],
            language_cnn_views=params['LANGUAGE_CNN_VIEWS'],
            language_max_pool_length=params['LANGUAGE_MAX_POOL_LENGTH'],
            output_dim=output_dim,
            visual_dim=visual_dim,
            mlp_hidden_dim=params['MLP_HIDDEN_SIZE'],
            merge_mode=params['MERGE_MODE'],
            multimodal_merge_mode=params['MULTIMODAL_MERGE_MODE'],
            num_mlp_layers=params['NUM_MLP_LAYERS'], 
            num_language_layers=params['NUM_LANGUAGE_LAYERS'], 
            mlp_activation=params['MLP_ACTIVATION'], 
            language_cnn_activation=params['LANGUAGE_CNN_ACTIVATION'],
            fusion_layer_index=params['FUSION_LAYER_INDEX'],
            is_go_backwards=params['IS_REVERSE_INPUT'],
            recurrent_encoder=recurrent_net.select[params['TEXT_ENCODER']], 
            recurrent_decoder=recurrent_net.select[params['TEXT_DECODER']],
            trainable_perception_name=params['TRAINABLE_PERCEPTION_NAME'],
            word_generator=model_zoo.word_generator[params['WORD_GENERATOR']],
            max_input_time_steps=params['MAX_INPUT_TIME_STEPS'],
            max_output_time_steps=params['MAX_OUTPUT_TIME_STEPS'],
            output_word_delimiter=train_dataset['answer_words_delimiter'])
    # building the model 
    model = model_zoo.select[params['MODEL']](model_config)
    model.create()
    #TODO: Doesn't work with very large models
    #model_picture(model=model, to_file=os.path.join('local', 
        #'model-{0}-{1}.png'.format(params['MODEL'], params['DATASET'])))
    if params['LR'] >= 0:
        current_optimizer = OPTIMIZERS[params['OPTIMIZER']](lr=params['LR'])
    else:
        current_optimizer = OPTIMIZERS[params['OPTIMIZER']]()
    model.compile(
            loss=params['LOSS'],
            optimizer=current_optimizer, 
            class_mode='categorical')
    # pickling the model
    """
    pickle_model(
            path=model_path, 
            model=model, 
            word2index_x=word2index_x,
            word2index_y=word2index_y,
            index2word_x=index2word_x,
            index2word_y=index2word_y)
    """
    #model_to_json(path=model_path, model=model)
    """
    if is_load_weights:
        start_era = era_weights_loader 
        model.load_weights(weights_path.format(
            epoch=start_era*params['MAX_EPOCH'],
            era=start_era))
        print('Restart the computations with weights from era {0}'.format(start_era))
        start_era += 1
    else:
        start_era = 0
    """
    ###

    ###
    # Training a model
    ###
    total_start_time = timer()
    total_number_of_epochs=params['MAX_EPOCH'] * params['MAX_ERA']

    if params['IS_VALIDATION_SET']:
        model.fit(train_input, Y, 
                batch_size=params['BATCH_SIZE'], 
                validation_data=validation_set,
                nb_epoch=total_number_of_epochs, 
                callbacks=callbacks,
                show_accuracy=True)
    elif params['VALIDATION_SPLIT'] > 0.0:
        model.fit(train_input, Y,
                batch_size=params['BATCH_SIZE'],
                validation_split=params['VALIDATION_SPLIT'],
                nb_epoch=total_number_of_epochs, 
                callbacks=callbacks,
                show_accuracy=True)
    else:
        model.fit(train_input, Y, 
                batch_size=params['BATCH_SIZE'], 
                nb_epoch=total_number_of_epochs, 
                callbacks=callbacks,
                show_accuracy=True)

    total_end_time = timer()
    time_difference = total_end_time - total_start_time
    print('In total is {0:.2f}s = {1:.2f}m'\
            .format(time_difference, time_difference/60.0))
    return True


if __name__ == '__main__':
    # setting up the input arguments
    args = parse_input_arguments()
    params = vars(args)

    print(params)

    main(params)

    print('Done!')

