"""
Parsing the input arguments.

Author: Mateusz Malinowski
Email: mmalinow@mpi-inf.mpg.de
"""

import argparse


###
# Default arguments
###
DEFAULT_MAX_INPUT_TIME_STEPS = 32
DEFAULT_MAX_OUTPUT_TIME_STEPS = 5 
DEFAULT_MAX_MEMORY_TIME_STEPS = 35
DEFAULT_TRUNCATE_OUTPUT_SPACE = 0
DEFAULT_TRUNCATE_INPUT_SPACE = 0
DEFAULT_NUM_TOP_QA_PAIRS=0
DEFAULT_MAX_ERA = 10
DEFAULT_MAX_EPOCH = 30
DEFAULT_BATCH_SIZE = 755
DEFAULT_HIDDEN_STATE_SIZE = 1000
DEFAULT_VISUAL_HIDDEN_STATE_SIZE = 1000
DEFAULT_TEXTUAL_EMBEDDING_SIZE = 1000
DEFAULT_VISUAL_EMBEDDING_SIZE = 1000
#DEFAULT_ADAPTATION_SIZE = 1000
DEFAULT_MLP_HIDDEN_SIZE = 1000
DEFAULT_NUM_MLP_LAYERS = 0
DEFAULT_NUM_LANGUAGE_LAYERS = 1
#DEFAULT_NUM_ADAPTATION_LAYERS = 1 
DEFAULT_TEMPERATURE = 0.001
DEFAULT_VALIDATION_SPLIT = 0.0
DEFAULT_TRAINING_SUBSET = -1
DEFAULT_VAL_SUBSET = -1
DEFAULT_TEST_SUBSET = -1
DEFAULT_REDUCE_RATE = 1.0
DEFAULT_MAX_NUMBER_REDUCTIONS = 10
DEFAULT_LR = -1
DEFAULT_LR_PATIENCE = 5
DEFAULT_FUSION_LAYER_INDEX = 0
DEFAULT_LANGUAGE_CNN_FILTERS = 1000
DEFAULT_LANGUAGE_CNN_FILTER_LENGTH = 3
DEFAULT_LANGUAGE_CNN_ACTIVATION = 'relu'
DEFAULT_LANGUAGE_CNN_VIEWS = 3
DEFAULT_LANGUAGE_MAX_POOL_LENGTH = 2
DEFAULT_VERBOSITY = ''
DEFAULT_WEIGHTS_LOADER_ERA = -1
DEFAULT_MERGE_MODE = 'ave'
DEFAULT_MULTIMODAL_MERGE_MODE = 'concat'
DEFAULT_WORD_REPRESENTATION = 'one_hot'
DEFAULT_OPTIMIZER = 'adam'
DEFAULT_TEXT_ENCODER = 'lstm'
DEFAULT_TEXT_DECODER = 'lstm'
DEFAULT_VISUAL_ENCODER = 'lstm'
DEFAULT_SEQUENCE_REDUCER = 'lstm'
DEFAULT_MEMORY_MATCH_ACTIVATION = 'softmax'
DEFAULT_MLP_ACTIVATION = 'relu'
DEFAULT_PERCEPTION='googlenet'
DEFAULT_PERCEPTION_LAYER='pool5-7x7_s1'
DEFAULT_PERCEPTION_SECOND_LAYER=''
DEFAULT_TRAINABLE_PERCEPTION_NAME='none'
DEFAULT_PARAMS='loss3_classifier'
DEFAULT_WORD_GENERATOR = 'max_likelihood'
DEFAULT_DATASET = 'daquar-triples'
DEFAULT_PARTS_EXTRACTOR = 'whole'
DEFAULT_MODEL = 'sequential-blind-single_answer'
DEFAULT_LOSS = 'categorical_crossentropy'
DEFAULT_METRIC = 'wups'
DEFAULT_VQA_ANSWER_MODE = 'single_random'
DEFAULT_PREDICTION_DATASET_FOLD = 'test'
DEFAULT_VISUALIZATION_URL = 'default'
DEFAULT_VISUALIZATION_FIG_LOSS_TITLE = 'Loss'
DEFAULT_VISUALIZATION_FIG_METRIC_TITLE = 'WUPS scores'
DEFAULT_WEIGHTS_LOADER_NAME = ''
DEFAULT_RESULTS_FILENAME = 'results'
DEFAULT_IS_REVERSE_INPUT=False
DEFAULT_IS_SAVE_WEIGHTS = False
DEFAULT_IS_LR_FIXED_REDUCTION = False
DEFAULT_IS_EARLY_STOPPING = False
DEFAULT_IS_VALIDATION_SET = False
DEFAULT_IS_ONLY_FIRST_ANSWER_WORD = True
DEFAULT_IS_WHOLE_ANSWER_AS_ANSWER_WORD = False


###
# Functions
###
def parse_input_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--max_input_time_steps', 
            dest='MAX_INPUT_TIME_STEPS', default=DEFAULT_MAX_INPUT_TIME_STEPS, type=int,
            help='Maximal number of time steps (word positions) in a question; ' +
            'by default {0}'.format(DEFAULT_MAX_INPUT_TIME_STEPS))
    arg_parser.add_argument('--max_output_time_steps',
            dest='MAX_OUTPUT_TIME_STEPS', default=DEFAULT_MAX_OUTPUT_TIME_STEPS, type=int,
            help='Maximal number of time steps (word positions) in an answer; ' +
            'by default {0}'.format(DEFAULT_MAX_OUTPUT_TIME_STEPS))
    arg_parser.add_argument('--max_visual_time_steps',
            dest='MAX_MEMORY_TIME_STEPS', default=DEFAULT_MAX_MEMORY_TIME_STEPS, type=int,
            help='Maximal number of memory time steps in the input; ' +
            'by default {0}'.format(DEFAULT_MAX_MEMORY_TIME_STEPS))
    arg_parser.add_argument('--truncate_output_space',
            dest='TRUNCATE_OUTPUT_SPACE', default=DEFAULT_TRUNCATE_OUTPUT_SPACE, type=int,
            help='Restrict the output space to the most frequent items if positive, otherwise all output items; ' +
            'by default {0}'.format(DEFAULT_TRUNCATE_OUTPUT_SPACE))
    arg_parser.add_argument('--truncate_input_space',
            dest='TRUNCATE_INPUT_SPACE', default=DEFAULT_TRUNCATE_INPUT_SPACE, type=int,
            help='Restrict the input space to the most frequent items if positive, otherwise all input items; ' +
            'by default {0}'.format(DEFAULT_TRUNCATE_INPUT_SPACE))
    arg_parser.add_argument('--number_most_frequent_qa_pairs',
            dest='NUM_TOP_QA_PAIRS', default=DEFAULT_NUM_TOP_QA_PAIRS, type=int,
            help='Restrict number of answers to K most frequent if positive, otherwise no restrictions; ' +
            'by default {0}'.format(DEFAULT_NUM_TOP_QA_PAIRS))
    arg_parser.add_argument('--max_era',
            dest='MAX_ERA', default=DEFAULT_MAX_ERA, type=int,
            help='Number of eras to loop over; one era contains many epochs; ' +
            'by default {0}'.format(DEFAULT_MAX_ERA))
    arg_parser.add_argument('--max_epoch',
            dest='MAX_EPOCH', default=DEFAULT_MAX_EPOCH, type=int,
            help='Number of epochs; by default {0}'.format(DEFAULT_MAX_EPOCH))
    arg_parser.add_argument('--batch_size',
            dest='BATCH_SIZE', default=DEFAULT_BATCH_SIZE, type=int,
            help='Number of samples in every batch; ' +
            'by default {0}'.format(DEFAULT_BATCH_SIZE))
    arg_parser.add_argument('--hidden_state_size',
            dest='HIDDEN_STATE_SIZE', default=DEFAULT_HIDDEN_STATE_SIZE, type=int,
            help='Size of the hidden state; by default {0}'.format(DEFAULT_HIDDEN_STATE_SIZE))
    arg_parser.add_argument('--visual_hidden_state_size',
            dest='VISUAL_HIDDEN_STATE_SIZE', default=DEFAULT_VISUAL_HIDDEN_STATE_SIZE, type=int,
            help='Size of the visual hidden state; by default {0}'.format(DEFAULT_VISUAL_HIDDEN_STATE_SIZE))
    arg_parser.add_argument('--textual_embedding_size',
            dest='TEXTUAL_EMBEDDING_SIZE', default=DEFAULT_TEXTUAL_EMBEDDING_SIZE, type=int,
            help='Size of the embedding layer; if 0 then no embedding is applied; by default {0}'.format(DEFAULT_TEXTUAL_EMBEDDING_SIZE))
    arg_parser.add_argument('--visual_embedding_size',
            dest='VISUAL_EMBEDDING_SIZE', default=DEFAULT_VISUAL_EMBEDDING_SIZE, type=int,
            help='Size of the visual embedding layer; by default {0}'.format(DEFAULT_VISUAL_EMBEDDING_SIZE))
    #arg_parser.add_argument('--adaptation_size',
            #dest='ADAPTATION_SIZE', default=DEFAULT_ADAPTATION_SIZE, type=int,
            #help='Size of the adaptation layer; by default {0}'.format(DEFAULT_ADAPTATION_SIZE))
    arg_parser.add_argument('--mlp_hidden_size',
            dest='MLP_HIDDEN_SIZE', default=DEFAULT_MLP_HIDDEN_SIZE, type=int,
            help='Size of the MLP layer; by default {0}'.format(DEFAULT_MLP_HIDDEN_SIZE))
    arg_parser.add_argument('--num_mlp_layers',
            dest='NUM_MLP_LAYERS', default=DEFAULT_NUM_MLP_LAYERS, type=int,
            help='Number of MLP layers; by default {0}'.format(DEFAULT_NUM_MLP_LAYERS))
    arg_parser.add_argument('--num_language_layers',
            dest='NUM_LANGUAGE_LAYERS', default=DEFAULT_NUM_LANGUAGE_LAYERS, type=int,
            help='Number of language layers; by default {0}'.format(DEFAULT_NUM_LANGUAGE_LAYERS))
    #arg_parser.add_argument('--num_adaptation_layers',
            #dest='NUM_ADAPTATION_LAYERS', default=DEFAULT_NUM_ADAPTATION_LAYERS, type=int,
            #help='Number of adaptation layers; by default {0}'.format(DEFAULT_NUM_ADAPTATION_LAYERS))
    arg_parser.add_argument('--temperature',
            dest='TEMPERATURE', default=DEFAULT_TEMPERATURE, type=float,
            help='Temperature for the predictions; the colder the more conservative (confident) answers; ' + 
            'by default {0}'.format(DEFAULT_TEMPERATURE))
    arg_parser.add_argument('--validation_split',
            dest='VALIDATION_SPLIT', default=DEFAULT_VALIDATION_SPLIT, type=float,
            help='Fraction of training data used for validation; by default {0}'.\
                    format(DEFAULT_VALIDATION_SPLIT))
    arg_parser.add_argument('--training_subset_size',
            dest='TRAINING_SUBSET_SIZE', default=DEFAULT_TRAINING_SUBSET, type=int,
            help='Size of the training subset, only for the monitoring if verbosity is set; ' + 
            'by default {0}'.format(DEFAULT_TRAINING_SUBSET))
    arg_parser.add_argument('--validation_subset_size',
            dest='VAL_SUBSET_SIZE', default=DEFAULT_VAL_SUBSET, type=int,
            help='Size of the validation subset, only for the monitoring if verbosity is set; ' + 
            'by default {0}'.format(DEFAULT_VAL_SUBSET))
    arg_parser.add_argument('--test_subset_size',
            dest='TEST_SUBSET_SIZE', default=DEFAULT_TEST_SUBSET, type=int,
            help='Size of the test subset, only for the monitoring if verbosity is set; ' + 
            'by default {0}'.format(DEFAULT_TEST_SUBSET))
    arg_parser.add_argument('--reduce_rate',
            dest='REDUCE_RATE', default=DEFAULT_REDUCE_RATE, type=float,
            help='Reduce learning rate; by default {0}'.format(DEFAULT_REDUCE_RATE))
    arg_parser.add_argument('--max_number_reductions',
            dest='MAX_NUMBER_REDUCTIONS', default=DEFAULT_MAX_NUMBER_REDUCTIONS, type=int,
            help='Maximal number of reductions; by default {0}'.format(DEFAULT_MAX_NUMBER_REDUCTIONS))
    arg_parser.add_argument('--lr',
            dest='LR', default=DEFAULT_LR, type=float,
            help='Learning rate if positive, otherwise default values for individual solvers are considered; by default {0}'.format(DEFAULT_LR))
    arg_parser.add_argument('--lr_patience',
            dest='LR_PATIENCE', default=DEFAULT_LR_PATIENCE, type=int,
            help='Patience (allowed number of epochs in stagnations); by default {0}'.format(DEFAULT_LR_PATIENCE))
    arg_parser.add_argument('--fusion_layer_index',
            dest='FUSION_LAYER_INDEX', default=DEFAULT_FUSION_LAYER_INDEX, type=int,
            help='Index of the language layer where the multimodal fusion happens; by default {0}'.format(DEFAULT_FUSION_LAYER_INDEX))
    arg_parser.add_argument('--language_cnn_filter_size',
            dest='LANGUAGE_CNN_FILTERS', default=DEFAULT_LANGUAGE_CNN_FILTERS, type=int,
            help='Number of filters for CNN language (dimensionality of the CNN output); by default {0}'.format(DEFAULT_LANGUAGE_CNN_FILTERS))
    arg_parser.add_argument('--language_cnn_filter_length',
            dest='LANGUAGE_CNN_FILTER_LENGTH', default=DEFAULT_LANGUAGE_CNN_FILTER_LENGTH, type=int,
            help='Size of receptive field of the language CNN filters; by default {0}'.format(DEFAULT_LANGUAGE_CNN_FILTER_LENGTH))
    arg_parser.add_argument('--language_cnn_activation',
            dest='LANGUAGE_CNN_ACTIVATION', default=DEFAULT_LANGUAGE_CNN_ACTIVATION, type=str,
            help='Activation for CNN language; by default {0}'.format(DEFAULT_LANGUAGE_CNN_ACTIVATION))
    arg_parser.add_argument('--language_cnn_views',
            dest='LANGUAGE_CNN_VIEWS', default=DEFAULT_LANGUAGE_CNN_VIEWS, type=int,
            help='Number of CNN views (e.g. two views are unigram and bigram); by default {0}'.format(DEFAULT_LANGUAGE_CNN_VIEWS))
    arg_parser.add_argument('--language_max_pool_length',
            dest='LANGUAGE_MAX_POOL_LENGTH', default=DEFAULT_LANGUAGE_MAX_POOL_LENGTH, type=int,
            help='Size of receptive field of max pooling; by default {0}'.format(DEFAULT_LANGUAGE_MAX_POOL_LENGTH))
    arg_parser.add_argument('--verbosity',
            dest='VERBOSITY', default=DEFAULT_VERBOSITY, type=str,
            help='Verbosity level with values separated by colon, there are: monitor_training_prediction; ' +
            'by default {0}'.format(DEFAULT_VERBOSITY))
    arg_parser.add_argument('--weights_loader_era',
            dest='WEIGHTS_LOADER_ERA', default=DEFAULT_WEIGHTS_LOADER_ERA, type=int,
            help='If non-negative then it sets the era to load the weights, otherwise no model is loaded; ' + 
            'by default {0}'.format(DEFAULT_WEIGHTS_LOADER_ERA))
    arg_parser.add_argument('--results_filename',
            dest='RESULTS_FILENAME', default=DEFAULT_RESULTS_FILENAME, type=str,
            help='Filename where the results of the predictions are stored; by default {0}'.format(DEFAULT_RESULTS_FILENAME))
    arg_parser.add_argument('--temporal_fusion',
            dest='MERGE_MODE', default=DEFAULT_MERGE_MODE, type=str,
            help='Temporal merging mode {0}'.format(DEFAULT_MERGE_MODE)) 
    arg_parser.add_argument('--multimodal_fusion',
            dest='MULTIMODAL_MERGE_MODE', default=DEFAULT_MULTIMODAL_MERGE_MODE, type=str,
            help='Multimodal merging mode {0}'.format(DEFAULT_MULTIMODAL_MERGE_MODE))
    arg_parser.add_argument('--word_representation',
            dest='WORD_REPRESENTATION', default=DEFAULT_WORD_REPRESENTATION, type=str,
            help='Word representation; by default {0}'.format(DEFAULT_WORD_REPRESENTATION))
    arg_parser.add_argument('--optimizer',
            dest='OPTIMIZER', default=DEFAULT_OPTIMIZER, type=str,
            help='Optimization algorithm for training; by default {0}'.format(DEFAULT_OPTIMIZER))
    arg_parser.add_argument('--text_encoder',
            dest='TEXT_ENCODER', default=DEFAULT_TEXT_ENCODER, type=str,
            help='Kind of used encoder; by default {0}'.format(DEFAULT_TEXT_ENCODER))
    arg_parser.add_argument('--text_decoder',
            dest='TEXT_DECODER', default=DEFAULT_TEXT_DECODER, type=str,
            help='Kind of used decoder, valid only in encoder-decoder architectures; ' + 
            'by default {0}'.format(DEFAULT_TEXT_DECODER))
    arg_parser.add_argument('--visual_encoder',
            dest='VISUAL_ENCODER', default=DEFAULT_VISUAL_ENCODER, type=str,
            help='Kind of used visual encoder, valid only in memory-based encoder-decoder architectures; ' + 
            'by default {0}'.format(DEFAULT_VISUAL_ENCODER))
    arg_parser.add_argument('--sequence_reducer',
            dest='SEQUENCE_REDUCER', default=DEFAULT_SEQUENCE_REDUCER, type=str,
            help='Kind of used sequence reducer, valid only in memory-based encoder-decoder architectures; ' + 
            'by default {0}'.format(DEFAULT_SEQUENCE_REDUCER))
    arg_parser.add_argument('--memory_match_activation',
            dest='MEMORY_MATCH_ACTIVATION', default=DEFAULT_MEMORY_MATCH_ACTIVATION, type=str,
            help='Kind of used memory match activation, valid only in memory-based architectures; ' + 
            'by default {0}'.format(DEFAULT_MEMORY_MATCH_ACTIVATION))
    arg_parser.add_argument('--mlp_activation',
            dest='MLP_ACTIVATION', default=DEFAULT_MLP_ACTIVATION, type=str,
            help='Kind of used MLP activation unit; by default {0}'.format(DEFAULT_MLP_ACTIVATION))
    arg_parser.add_argument('--perception',
            dest='PERCEPTION', default=DEFAULT_PERCEPTION, type=str,
            help='Kind of a pre-trained perception model used; by default {0}'.format(DEFAULT_PERCEPTION))
    arg_parser.add_argument('--perception_layer',
            dest='PERCEPTION_LAYER', default=DEFAULT_PERCEPTION_LAYER, type=str,
            help='Kind of layer in the pre-trained perception used; by default {0}'.format(DEFAULT_PERCEPTION_LAYER))
    arg_parser.add_argument('--perception_second_layer',
            dest='PERCEPTION_SECOND_LAYER', default=DEFAULT_PERCEPTION_SECOND_LAYER, type=str,
            help='Kind of layer in the pre-trained perception used; by default {0}'.format(DEFAULT_PERCEPTION_SECOND_LAYER))
    arg_parser.add_argument('--trainable_perception',
            dest='TRAINABLE_PERCEPTION_NAME', default=DEFAULT_TRAINABLE_PERCEPTION_NAME, type=str,
            help='Perception that is used to train or fine-tune, or none if we want to rely on a pre-trained perception; '\
                    + ' by default {0}'.format(DEFAULT_TRAINABLE_PERCEPTION_NAME))
    arg_parser.add_argument('--params',
            dest='PARAMS', default=DEFAULT_PARAMS, type=str,
            help='Kind of params in the perception used; by default {0}'.format(DEFAULT_PARAMS))
    arg_parser.add_argument('--word_generator',
            dest='WORD_GENERATOR', default=DEFAULT_WORD_GENERATOR, type=str,
            help='Procedure to generate single words; ' + 
            'by default {0}'.format(DEFAULT_WORD_GENERATOR))
    arg_parser.add_argument('--dataset',
            dest='DATASET', default=DEFAULT_DATASET, type=str,
            help='Kind of used dataset; by default {0}'.format(DEFAULT_DATASET))
    arg_parser.add_argument('--parts_extractor',
            dest='PARTS_EXTRACTOR', default=DEFAULT_PARTS_EXTRACTOR, type=str,
            help='Kind of parts extractor; only if image parts are concerned; by default {0}'.format(DEFAULT_PARTS_EXTRACTOR))
    arg_parser.add_argument('--model',
            dest='MODEL', default=DEFAULT_MODEL, type=str,
            help='Kind of used model; by default {0}'.format(DEFAULT_MODEL))
    arg_parser.add_argument('--loss',
            dest='LOSS', default=DEFAULT_LOSS, type=str,
            help='Kind of used loss; by default {0}'.format(DEFAULT_LOSS))
    arg_parser.add_argument('--metric',
            dest='METRIC', default=DEFAULT_METRIC, type=str,
            help='Kind of used metric; by default {0}'.format(DEFAULT_METRIC))
    arg_parser.add_argument('--vqa_answer_mode',
            dest='VQA_ANSWER_MODE', default=DEFAULT_VQA_ANSWER_MODE, type=str,
            help='VQA answer mode; by default {0}'.format(DEFAULT_VQA_ANSWER_MODE))
    arg_parser.add_argument('--prediction_dataset_fold',
            dest='PREDICTION_DATASET_FOLD', default=DEFAULT_PREDICTION_DATASET_FOLD, type=str,
            help='Dataset chosen for predictions; by default {0}'.format(DEFAULT_PREDICTION_DATASET_FOLD))
    arg_parser.add_argument('--visualization_url',
            dest='VISUALIZATION_URL', default=DEFAULT_VISUALIZATION_URL, type=str,
            help='Bokeh url; by default {0}'.format(DEFAULT_VISUALIZATION_URL))
    arg_parser.add_argument('--visualization_fig_loss_title',
            dest='VISUALIZATION_FIG_LOSS_TITLE', default=DEFAULT_VISUALIZATION_FIG_LOSS_TITLE, type=str,
            help='Bokeh loss figure title; by default {0}'.format(DEFAULT_VISUALIZATION_FIG_LOSS_TITLE))
    arg_parser.add_argument('--visualization_fig_metric_title',
            dest='VISUALIZATION_FIG_METRIC_TITLE', default=DEFAULT_VISUALIZATION_FIG_METRIC_TITLE, type=str,
            help='Bokeh metric figure title; by default {0}'.format(DEFAULT_VISUALIZATION_FIG_METRIC_TITLE))
    arg_parser.add_argument('--weights_loader_name',
            dest='WEIGHTS_LOADER_NAME', default=DEFAULT_WEIGHTS_LOADER_NAME, type=str,
            help='The main name for the weights loader; by default {0}'.format(DEFAULT_WEIGHTS_LOADER_NAME))
    # boolean arguments
    arg_parser.add_argument('--reverse_input',
            dest='IS_REVERSE_INPUT', action='store_true',
            help='If it is set up then the input is processed in a reverse order ' +
            'by default {0}'.format('--reverse_input' if DEFAULT_IS_REVERSE_INPUT else 'no_reverse_input'))
    arg_parser.add_argument('--no_reverse_input',
            dest='IS_REVERSE_INPUT', action='store_false',
            help='If it is set up then the input is processed in a reverse order ' +
            'by default {0}'.format('--reverse_input' if DEFAULT_IS_REVERSE_INPUT else 'no_reverse_input'))
    arg_parser.set_defaults(IS_REVERSE_INPUT=DEFAULT_IS_REVERSE_INPUT)
    arg_parser.add_argument('--store_weights',
            dest='IS_SAVE_WEIGHTS', action='store_true',
            help='If it is set up then the weights are saved in each era; ' +
            'by default {0}'.format('store_weights' if DEFAULT_IS_SAVE_WEIGHTS else 'no_store_weights'))
    arg_parser.add_argument('--no_store_weights',
            dest='IS_SAVE_WEIGHTS', action='store_false',
            help='If it is set up then the weights are forgotten; ' +
            'by default {0}'.format('store_weights' if DEFAULT_IS_SAVE_WEIGHTS else 'no_store_weights'))
    arg_parser.set_defaults(IS_SAVE_WEIGHTS=DEFAULT_IS_SAVE_WEIGHTS)
    arg_parser.add_argument('--lr_fixed_reduction',
            dest='IS_LR_FIXED_REDUCTION', action='store_true',
            help='If it is set up early stopping is applied based on val acc; ' +
            'by default {0}'.format('early_stopping' if DEFAULT_IS_LR_FIXED_REDUCTION else 'no_early_stopping'))
    arg_parser.add_argument('--no_lr_fixed_reduction',
            dest='IS_LR_FIXED_REDUCTION', action='store_false',
            help='If it is set up early stopping is applied based on val acc; ' +
            'by default {0}'.format('early_stopping' if DEFAULT_IS_LR_FIXED_REDUCTION else 'no_early_stopping'))
    arg_parser.set_defaults(IS_LR_FIXED_REDUCTION=DEFAULT_IS_LR_FIXED_REDUCTION)
    arg_parser.add_argument('--early_stopping',
            dest='IS_EARLY_STOPPING', action='store_true',
            help='If it is set up early stopping is applied based on val acc; ' +
            'by default {0}'.format('early_stopping' if DEFAULT_IS_EARLY_STOPPING else 'no_early_stopping'))
    arg_parser.add_argument('--no_early_stopping',
            dest='IS_EARLY_STOPPING', action='store_false',
            help='If it is set up early stopping is applied based on val acc; ' +
            'by default {0}'.format('early_stopping' if DEFAULT_IS_EARLY_STOPPING else 'no_early_stopping'))
    arg_parser.set_defaults(IS_EARLY_STOPPING=DEFAULT_IS_EARLY_STOPPING)
    arg_parser.add_argument('--use_validation',
            dest='IS_VALIDATION_SET', action='store_true',
            help='If it is set up then the validation set is used; ' + 
            'by default {0}'.format('use_validation' if DEFAULT_IS_VALIDATION_SET else 'no_validation'))
    arg_parser.add_argument('--no_validation',
            dest='IS_VALIDATION_SET', action='store_false',
            help='If it is set up then there is no validation set; ' +
            'by default {0}'.format('use_validation' if DEFAULT_IS_VALIDATION_SET else 'no_validation'))
    arg_parser.set_defaults(IS_VALIDATION_SET=DEFAULT_IS_VALIDATION_SET)
    arg_parser.add_argument('--use_first_answer_words',
            dest='IS_ONLY_FIRST_ANSWER_WORD', action='store_true',
            help='If it is set up then first answer words are considered (otherwise, all); ' + 
            'by default {0}'.format('use_first_answer_words' if DEFAULT_IS_ONLY_FIRST_ANSWER_WORD else 'use_all_answer_words'))
    arg_parser.add_argument('--use_all_answer_words',
            dest='IS_ONLY_FIRST_ANSWER_WORD', action='store_false',
            help='If it is set up then all answer words are considered (otherwise, only the first); ' +
            'by default {0}'.format('use_first_answer_words' if DEFAULT_IS_ONLY_FIRST_ANSWER_WORD else 'use_all_answer_words'))
    arg_parser.set_defaults(IS_ONLY_FIRST_ANSWER_WORD=DEFAULT_IS_ONLY_FIRST_ANSWER_WORD)
    arg_parser.add_argument('--use_whole_answer_as_answer_word',
            dest='IS_WHOLE_ANSWER_AS_ANSWER_WORD', action='store_true',
            help='If it is set up then one answer words is the whole answer; ' +
            'by default {0}'.format('answer word is the whole answer' if DEFAULT_IS_WHOLE_ANSWER_AS_ANSWER_WORD else 'split answer into answer words'))
    arg_parser.add_argument('--split_answer_into_answer_words',
            dest='IS_WHOLE_ANSWER_AS_ANSWER_WORD', action='store_false',
            help='If it is set up then one answer words is the whole answer; ' +
            'by default {0}'.format('answer word is the whole answer' if DEFAULT_IS_WHOLE_ANSWER_AS_ANSWER_WORD else 'split answer into answer words'))
    arg_parser.set_defaults(IS_WHOLE_ANSWER_AS_ANSWER_WORD=DEFAULT_IS_WHOLE_ANSWER_AS_ANSWER_WORD)
    # not-working arguments
    #arg_parser.add_argument('--gpu_core',
            #dest='GPU_CORE', default=-1, type=int,
            #help='GPU Core, if -1 then the core is read from the config file')
    args = arg_parser.parse_args()

    return args

