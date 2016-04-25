#!/usr/bin/env python
from __future__ import unicode_literals
from __future__ import print_function

"""
Builds input/output space.

Author: Mateusz Malinowski
Email: mmalinow@mpi-inf.mpg.de
"""

import numpy as np

from toolz import itemmap


__all__ = ['build_vocabulary', 'index_sequence',
        'encode_questions_index','encode_questions_one_hot',
        'encode_answers_one_hot']

###
###
# Constants
###
PADDING = '<pad>'
UNKNOWN = '<unk>'
EOA = '<eoa>'       # end of answer
EOQ = '<eoq>'       # end of question
EXTRA_WORDS_NAMES = [PADDING, UNKNOWN, EOA, EOQ]
EXTRA_WORDS = {PADDING:0, UNKNOWN:1, EOA:2, EOQ:3}
EXTRA_WORDS_ID = itemmap(reversed, EXTRA_WORDS)

###
# Functions
###
def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@static_vars(counter=len(EXTRA_WORDS))
def _myinc(d):
    """
    Gets a tuple d, and returns d[0]: id.
    """
    x = d[0]
    _myinc.counter += 1
    return (x, _myinc.counter - 1)


def build_vocabulary(this_wordcount, extra_words=EXTRA_WORDS, 
        is_reset=True, truncate_to_most_frequent=0):
    """
    Builds vocabulary from wordcount.
    It also adds extra words to the vocabulary.

    In:
        this_wordcount - dictionary of wordcounts, e.g. {'cpu':3}
        extra_words - additional words to build the vocabulary
            dictionary of {word: id}
            by default {UNKNOWN: 0}
        is_reset - if True we restart the vocabulary counting
            by defaults False
        truncate_to_most_frequent - if positive then the vocabulary
            is truncated to 'truncate_to_most_frequent' words;
            by default 0 

    Out:
        word2index - mapping from words to indices
        index2word - mapping from indices to words
    """
    if is_reset:
        _myinc.counter=len(EXTRA_WORDS)
    if truncate_to_most_frequent > 0:
        sorted_wordcount = dict(sorted(
                this_wordcount.items(), key=lambda x:x[1], reverse=True)[:truncate_to_most_frequent])
        this_wordcount = sorted_wordcount
        
    word2index = itemmap(_myinc, this_wordcount)
    if not extra_words == {}:
        assert(all([el not in word2index.values() for el in extra_words.values()]))
        word2index.update(extra_words)
    index2word = itemmap(reversed, word2index)
    return word2index, index2word


def index_sequence(x, word2index):
    """
    Converts list of words into a list of its indices wrt. word2index, that is into
    index encoded sequence.

    In:
        x - list of lines
        word2index - mapping from words to indices

    Out:
        a list of the list of indices that encode the words
    """
    one_hot_x = []
    for line in x:
        line_list = []
        for w in line.split():
            w = w.strip()
            if w in word2index: this_ind = word2index[w]
            else: this_ind = word2index[UNKNOWN]
            line_list.append(this_ind)
        one_hot_x.append(line_list)
    return one_hot_x


def encode_questions_index(x, word2index_x, max_time_steps=None):
    """
    Index-based encoding of questions.

    In:
        x - list of questions
        word2index_x - mapping from question words to indices (inverted vocabulary)
        max_time_steps - maximal number of words in the question (max. time steps);
            if None then all question words are taken;
            by default None 
    Out:
        a list of encoded questions
    """
    x_modified = [q + ' ' + EOQ for q in x]
    if max_time_steps is not None:
        x_modified = [' '.join(q.split()[:max_time_steps]) for q in x]
    return index_sequence(x_modified, word2index_x)


def encode_questions_one_hot(x, word2index_x, max_time_steps):
    """
    One-hot encoding of questions.

    In:
        x - list of  questions
        word2index_x - mapping from question words to indices (inverted vocabulary)
        max_time_steps - maximal number of words in the sequence (max. time steps)

    Out:
        boolean tensor of size: data_size x max_time_steps x vocabulary_size
            for a given question and a time step there is only one '1'
    """
    X = np.zeros((len(x), max_time_steps, len(word2index_x.keys())), 
            dtype=np.bool)
    # encode questions
    for question_no, question in enumerate(x):
        question_word_list = question.split()
        question_word_list.append(EOQ)
        for word_no, word in enumerate(question_word_list):
            word = word.strip()
            if word_no == max_time_steps - 1:
                # we need to finish
                this_index = word2index_x[EOQ]
            else:
                if word in word2index_x:
                    this_index = word2index_x[word]
                else:
                    this_index = word2index_x[UNKNOWN]
            X[question_no, word_no, this_index] = 1
    return X

def encode_questions_dense(x, word_encoder, max_time_steps, 
        is_remove_question_symbol=False):
    """
    Dense representation of questions.

    In:
        x - list of questions
        word_encoder - encodes words
        max_time_steps - maximal number of words in the sequence (max. time steps)
        is_remove_question_symbol - true if we remove question symbols from the questions;
            by default it is False

    Out:
        float tensor of size: data_size x max_time_steps x dense_encoding_size
    """
    word_encoder_dim = word_encoder(unicode(x[0].split()[0].strip())).vector.shape[0]
    X = np.zeros((len(x), max_time_steps, word_encoder_dim))
    for question_no, question in enumerate(x):
        question_word_list = question.split()
        if is_remove_question_symbol and question_word_list[-1] == '?':
            question_word_list = question_word_list[:-1]
        reversed_question_word_list = question_word_list[::-1]
        for word_no, raw_word in enumerate(reversed_question_word_list):
            word = unicode(raw_word.strip())
            this_representation = word_encoder(word).vector
            if max_time_steps - word_no - 1 >= 0:
                X[question_no, max_time_steps - word_no - 1, :] = this_representation
            else:
                break
    return X


def encode_answers_one_hot(y, word2index_y, max_answer_time_steps=10, 
        is_only_first_answer_word=False, answer_words_delimiter=','):
    """
    One-hot encoding of answers.
    If more than first answer word is encoded then the answer words 
    are modelled as sequence.

    In:
        y - list of answers
        word2index_y - mapping from answer words to indices (vocabulary)
        max_answer_time_steps - maximal number of words in the sequence (max. time steps)
            by default 10
        is_only_first_answer_word - if True then only first answer word is taken
            by default False
        answer_words_delimiter - a symbol for splitting answer into answer words;
            if None is provided then we don't split answer into answer words 
            (that is the whole answer is an answer word);
            by default ','

    Out:
        Y - boolean matrix of size: 
                data_size x vocabulary_size if there is only single answer word
                data_size x max_answer_time_steps x vocabulary_size otherwise
                    the matrix is padded
            for a given answer and a time step there is only one '1'
        y_gt - list of answers
            the same as input 'y' if is_only_first_answer_word==False
            only first words from 'y' if is_only_first_answer_word==True
    """
    # encode answers
    if is_only_first_answer_word:
        Y = np.zeros((len(y), len(word2index_y.keys())), dtype=np.bool)
        y_gt = []
    else:
        Y = np.zeros((len(y), max_answer_time_steps, len(word2index_y.keys())),
                dtype=np.bool)
        y_gt = y

    if answer_words_delimiter is None:
        assert(is_only_first_answer_word==True)

    for answer_no, answer in enumerate(y):
        if answer_words_delimiter is not None:
            answer_split = answer.split(answer_words_delimiter)
        else:
            answer_split = [answer]
        for word_no, word in enumerate(answer_split):
            word = word.strip()
            if is_only_first_answer_word:
                y_gt.append(word)
                if word in word2index_y:
                    Y[answer_no, word2index_y[word]] = 1
                else:
                    Y[answer_no, word2index_y[UNKNOWN]] = 1
                break
            else:
                if word_no == max_answer_time_steps - 1:
                    break
                if word in word2index_y:
                    Y[answer_no, word_no, word2index_y[word]] = 1
                else:
                    Y[answer_no, word_no, word2index_y[UNKNOWN]] = 1
        if not is_only_first_answer_word:
            Y[answer_no, 
                    min(len(answer_split), max_answer_time_steps-1), 
                    word2index_y[EOA]] = 1
    return Y, y_gt


def shift(X, new_vector=None, time_axis=1):
    """
    Shifts input X along time_axis by one. 
    At the new place it introduces new_word_id.
    The method doesn't change the size of X, so 
    the last column along time axis is forgotten.

    In:
        X - input array;
            X has to have one more dimension than time_axis,
            so if time_axis == 1 then X has 3 dimensions (0,1,2)
        new_vector - new vector that replaces the column at time axis;
            if None, then the last column is added at the first position;
            by default None
        time_axis - axis where shifting happens
    Out:
        shifted version of X along the time axis
    """
    tmp = np.roll(X, 1, time_axis)
    if new_vector is None:
        return tmp
    if time_axis==0:
        tmp[0,:] = new_vector 
    elif time_axis==1:
        tmp[:,0,:] = new_vector 
    elif time_axis==2:
        tmp[:,:,0,:] = new_vector 
    elif time_axis==3:
        tmp[:,:,:,0,:] = new_vector 
    else:
        raise NotImplementedError
    return tmp


def shift_with_index_vector(X, index, size, time_axis, value=1, dtype=np.bool):
    """
    Shifts X along time_axis, and inserts a one-hot vector at the first 
    column at this axis.

    In:
        X - n-array
        index - index for value, 
            the other elements of the corresponding vector are 0
        time_axis - axis where shifting happens
        value - value to place at index;
            by default 1
        dtype - type of the new vector;
            by default np.bool
    """
    tmp = np.zeros(size, dtype=dtype)
    tmp[..., index] = value
    return shift(X, tmp, time_axis)


