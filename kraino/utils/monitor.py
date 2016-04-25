#!/usr/bin/env python
from __future__ import print_function

"""
Monitoring tools.

Author: Mateusz Malinowski
Email: mmalinow@mpi-inf.mpg.de
"""

from .read_write import list2file


def _dirac(pred, gt):
    return int(pred==gt)


def print_qa(questions, answers_gt, answers_gt_original, answers_pred, 
        era, similarity=_dirac, path=''):
    """
    In:
        questions - list of questions
        answers_gt - list of answers (after modifications like truncation)
        answers_gt_original - list of answers (before modifications)
        answers_pred - list of predicted answers
        era - current era
        similarity - measure that measures similarity between gt_original and prediction;
            by default dirac measure
        path - path for the output (if empty then stdout is used)
            by fedault an empty path
    Out:
        the similarity score
    """
    assert(len(questions)==len(answers_gt))
    assert(len(questions)==len(answers_pred))
    output=['-'*50, 'Era {0}'.format(era)]
    score = 0.0
    for k, q in enumerate(questions):
        a_gt=answers_gt[k]
        a_gt_original=answers_gt_original[k]
        a_p=answers_pred[k]
        score += _dirac(a_p, a_gt_original)
        output.append('question: {0}\nanswer: {1}\nanswer_original: {2}\nprediction: {3}\n'\
                .format(q, a_gt, a_gt_original, a_p))
    score = (score / len(questions))*100.0
    output.append('Score: {0}'.format(score))
    if path == '':
        print('%s' % '\n'.join(map(str, output)))
    else:
        list2file(path, output)
    return score

