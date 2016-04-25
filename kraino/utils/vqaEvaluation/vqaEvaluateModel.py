#!/usr/bin/env python
# coding: utf-8

"""
Slightly modified variant of the original script.

Author of the original: Aishwarya Agrawal
"""

import sys
dataDir = '/BS/databases/vqa_1.0/VQA'
sys.path.insert(0, '../vqaTools')
from vqa import VQA
from vqaClassNormalizedEval import VQAClassNormalizedEval as VQAEval
import matplotlib.pyplot as plt
import skimage.io as io
import json
import random
import os

if len(sys.argv) != 4:
    print 'Usage: python vqaEvaluateModel datasetFold resultType isVisualisation'
    print 'E.g.: python vqaEvaluateModel val image_bow False'
    sys.exit(1)

datasetFold = sys.argv[1]
resultType  = sys.argv[2]
if sys.argv[3] == 'True':
    isVisualisation = True
elif sys.argv[3] == 'False':
    isVisualisation = False
else:
    raise NotImplementedError()

# set up file names and paths
taskType    ='OpenEnded'
dataType    ='mscoco'  # 'mscoco' for real and 'abstract_v002' for abstract
if datasetFold == 'train':
    dataSubType ='train2014' 
elif datasetFold == 'val':
    dataSubType = 'val2014'
else:
    raise NotImplementedError()
annFile     ='%s/Annotations/%s_%s_annotations.json'%(dataDir, dataType, dataSubType)
quesFile    ='%s/Questions/%s_%s_%s_questions.json'%(dataDir, taskType, dataType, dataSubType)
imgDir      ='%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)
fileTypes   = ['results', 'accuracy', 'evalQA', 'evalQuesType', 'evalAnsType'] 

# An example result json file has been provided in './Results' folder.  

[resFile, accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = \
    ['../../../local/results/%s.%s.%s.%s.%s.json'%(taskType, dataType, dataSubType, resultType, fileType) for fileType in fileTypes]  

# create vqa object and vqaRes object
vqa = VQA(annFile, quesFile)
vqaRes = vqa.loadRes(resFile, quesFile)
# create vqaEval object by taking vqa and vqaRes
vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2

# evaluate results
"""
If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
By default it uses all the question ids in annotation file
"""
vqaEval.evaluate() 
# print accuracies
print "\n"
print "Per Question Type Accuracy is the following:"
for quesType in vqaEval.accuracy['perQuestionType']:
	print "%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType])
print "Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall'])
print "\n"
print "Overall per class accuracy is %.02f\n" %(vqaEval.accuracy['classNormalizedOverall'])
print "\n"
print "Per Answer Type Accuracy is the following:"
for ansType in vqaEval.accuracy['perAnswerType']:
	print "%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType])
print "\n"
# demo how to use evalQA to retrieve low score result
if isVisualisation == True:
    evals = [quesId for quesId in vqaEval.evalQA if vqaEval.evalQA[quesId]<35]   #35 is per question percentage accuracy
    if len(evals) > 0:
        print 'ground truth answers'
        randomEval = random.choice(evals)
        randomAnn = vqa.loadQA(randomEval)
        vqa.showQA(randomAnn)

        print '\n'
        print 'generated answer (accuracy %.02f)'%(vqaEval.evalQA[randomEval])
        ann = vqaRes.loadQA(randomEval)[0]
        print "Answer:   %s\n" %(ann['answer'])

        imgId = randomAnn[0]['image_id']
        imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
        if os.path.isfile(imgDir + imgFilename):
            I = io.imread(imgDir + imgFilename)
            plt.imshow(I)
            plt.axis('off')
            plt.show()

    # plot accuracy for various question types
    plt.bar(range(len(vqaEval.accuracy['perQuestionType'])), vqaEval.accuracy['perQuestionType'].values(), align='center')
    plt.xticks(range(len(vqaEval.accuracy['perQuestionType'])), vqaEval.accuracy['perQuestionType'].keys(), rotation='0',fontsize=10)
    plt.title('Per Question Type Accuracy', fontsize=10)
    plt.xlabel('Question Types', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.show()

# save evaluation results to ./Results folder
json.dump(vqaEval.accuracy,                 open(accuracyFile,      'w'))
json.dump(vqaEval.evalQA,                   open(evalQAFile,        'w'))
json.dump(vqaEval.evalQuesType,             open(evalQuesTypeFile,  'w'))
json.dump(vqaEval.evalAnsType,              open(evalAnsTypeFile,   'w'))

