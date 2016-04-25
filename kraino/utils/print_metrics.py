#!/usr/bin/env python
from __future__ import print_function

"""
Selects and prints metrics.

Author: Mateusz Malinowski
Email: mmalinow@mpi-inf.mpg.de
"""

import os

from uuid import uuid4 

from compute_wups import get_metric_score as wups_score
from compute_wups import get_class_metric_score as class_wups_score
from data_provider import vqa_save_results as vqa_store
from vqaEvaluation.vqaClassNormalizedEval import VQAClassNormalizedEval as VQAEval


def average_over_dictionary(mydict):
    """
    Average over dictionary values.
    """
    ave = sum([x for x in mydict.values()])/len(mydict)
    return ave


def show_wups(gt_list, pred_list, verbose, extra_vars):
    """
    In:
        gt_list - ground truth list 
        pred_list - list of predictions
        verbose - if greater than 0 the metric measures are printed out
        extra_vars - not used here

    Out:
        list of key, value pairs (dict) such that 
        'value' denotes the performance number
        and 'name' denotes the name of the metric
    """
    acc = wups_score(gt_list, pred_list, -1) * 100.0
    wups_at_09 = wups_score(gt_list, pred_list, 0.9) * 100.0
    #wups_at_0 = wups_score(gt_list, pred_list, 0.0) * 100.0
    wups_at_0 = -1.0
    per_class_acc_tmp = class_wups_score(gt_list, pred_list, -1)
    #per_class_wups_at_09_tmp = class_wups_score(gt_list, pred_list, 0.9)
    per_class_wups_at_09_tmp = None
    per_class_acc = {k:v*100.0 for k,v in per_class_acc_tmp.items()}
    if per_class_wups_at_09_tmp is not None:
        per_class_wups_at_09 = {k:v*100.0 for k,v in per_class_wups_at_09_tmp.items()}
    else:
        per_class_wups_at_09 = None
    class_acc = average_over_dictionary(per_class_acc_tmp)*100.0 
    if per_class_wups_at_09_tmp is not None:
        class_wups_at_09 = average_over_dictionary(per_class_wups_at_09_tmp)*100.0
    else:
        class_wups_at_09 = -1.0
    class_wups_at_0 = -1.0
    if verbose > 0:
        print('METRIC: Accuracy is {0}, wups at 0.9 is {1}, wups at 0.0 is {2}'.format(
            acc, wups_at_09, wups_at_0))
        print('CLASS METRIC: Accuracy is {0}, wups at 0.9 is {1}, wups at 0.0 is {2}'.format(
            class_acc, class_wups_at_09, class_wups_at_0))
    return [{'value':acc, 'name':'accuracy'},
            {'value':wups_at_09, 'name':'wups at 0.9'}, 
            {'value':wups_at_0, 'name':'wups at 0.0'},
            {'value':per_class_acc, 'name':'per class accuracy',
                'idiosyncrasy':'long:muted'},
            {'value':per_class_wups_at_09, 'name':'per class wups at 0.9',
                'idiosyncrasy':'long:muted'},
            {'value':class_acc, 'name':'class accuracy'},
            {'value':class_wups_at_09, 'name':'class wups at 0.9'},
            {'value':class_wups_at_0, 'name':'class wups at 0'},]

def show_vqa(gt_list, pred_list, verbose, extra_vars):
        #question_id, vqa_object, 
        #dataset_root=None):
    """
    In:
        gt_list - ground truth list 
        pred_list - list of predictions
        verbose - if greater than 0 the metric measures are printed out
        extra_vars - extra variables, here are:
            extra_vars['vqa'] - the vqa object
            extra_vars['resfun'] - function from the results file to the vqa object
            extra_vars['question_id'] - list of the question ids

    Out:
        list of key, value pairs (dict) such that 
        'value' denotes the performance number
        and 'name' denotes the name of the metric
    """
    # TODO: quite hacky way of creating and next reading the file
    if verbose > 0:
        print('dumping json file ...')
    vqa_object = extra_vars['vqa_object']
    results_path = '/tmp/vqa_metric_{0}.json'.format(uuid4())
    #print(results_path)
    vqa_store(extra_vars['question_id'], pred_list, results_path)
    vqa_res = extra_vars['resfun'](results_path)
    os.remove(results_path)
    if verbose > 0:
        print('dumping finished')
    ### 
    vqaEval = VQAEval(vqa_object, vqa_res, n=2)
    vqaEval.evaluate()
    acc_overall = vqaEval.accuracy['overall']
    acc_yes_no = vqaEval.accuracy['perAnswerType']['yes/no']
    acc_number = vqaEval.accuracy['perAnswerType']['number']
    acc_other = vqaEval.accuracy['perAnswerType']['other']
    acc_per_class = vqaEval.accuracy['perAnswerClass']
    acc_class_normalized = vqaEval.accuracy['classNormalizedOverall']

    if verbose > 0:
        print('METRIC: Accuracy yes/no is {0}, other is {1}, number is {2}, overall is {3}, class normalized is {4}'.\
                format(acc_yes_no, acc_other, acc_number, acc_overall, acc_class_normalized))
    return [{'value':acc_overall, 'name':'overall accuracy'},
            {'value':acc_yes_no, 'name':'yes/no accuracy'},
            {'value':acc_number, 'name':'number accuracy'},
            {'value':acc_other, 'name':'other accuracy'},
            {'value':acc_class_normalized, 'name':'class accuracy'},
            {'value':acc_per_class, 'name':'per answer class', 
                'idiosyncrasy':'long:muted'},]


select = {
        'wups' : show_wups,
        'vqa' : show_vqa
        }

