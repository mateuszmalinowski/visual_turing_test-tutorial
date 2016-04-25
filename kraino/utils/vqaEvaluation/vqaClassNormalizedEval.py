# coding=utf-8

"""
Extension of 'Agrawals's vqa evaluation script' with a class-specific metric.

Mateusz Malinowski [mmalinow@mpi-inf.mpg.de]
"""

import numpy as np

from toolz import frequencies

from vqaEval import VQAEval


class VQAClassNormalizedEval(VQAEval):
    """
    A class normalized evaluation metric. 

    It assignes to every answer its answer class, and next assigs the answer
    class to accuracies.
    It does the assignment based on the most frequent answers.
    """
    def __init__(self, vqa, vqaRes, n=2):
        VQAEval.__init__(self, vqa, vqaRes, n)

        print "Initialize class normalized evaluation..."
        # calculates answer frequencies over the current answers (train, val,
        # etc.)
        quesIds = [x for x in self.params['question_id']]
        gts = {}
        for quesId in quesIds:
            gts[quesId] = self.vqa.qa[quesId]

        # consider frequencies for all answers
        all_answers = [x['answer'] for y in gts for x in gts[y]['answers']]
        self.answer2freq = frequencies(all_answers)
        print "Class normalized evaluation initialized!"

    def evaluate(self, quesIds=None):
        if quesIds == None:
            quesIds = [quesId for quesId in self.params['question_id']]
        gts = {}
        res = {}
        for quesId in quesIds:
            gts[quesId] = self.vqa.qa[quesId]
            res[quesId] = self.vqaRes.qa[quesId]
       
        # =================================================
        # Compute accuracy
        # =================================================
        accQA       = []
        accQuesType = {}
        accAnsType  = {}
        accAnswerClass = {}
        print "computing accuracy"
        step = 0
        
        for quesId in quesIds:
            resAns      = res[quesId]['answer']
            resAns      = resAns.replace('\n', ' ')
            resAns      = resAns.replace('\t', ' ')
            resAns      = resAns.strip()
            resAns      = self.processPunctuation(resAns)
            resAns      = self.processDigitArticle(resAns)
            gtAcc  = []
            gtAnswers = [ans['answer'] for ans in gts[quesId]['answers']]

            # ============================================
            # Create the abstract classes
            # ============================================
            # take confident answers if possible
            gtAnswersConfident = [ans['answer'] for ans in gts[quesId]['answers'] if ans['answer_confidence'] == 'yes'] 
            if gtAnswersConfident == []:
                gtAnswersConfident = gtAnswers
            sortedGtAnswers_y = sorted(gtAnswersConfident)
            sortedGtAnswers_x = map(lambda x:self.answer2freq[x], sortedGtAnswers_y)
            answerClass = sortedGtAnswers_y[np.argmax(sortedGtAnswers_x)]
            # ============================================
            if len(set(gtAnswers)) > 1: 
                for ansDic in gts[quesId]['answers']:
                    ansDic['answer'] = self.processPunctuation(ansDic['answer'])
            for gtAnsDatum in gts[quesId]['answers']:
                otherGTAns = [item for item in gts[quesId]['answers'] if item!=gtAnsDatum]
                matchingAns = [item for item in otherGTAns if item['answer']==resAns]
                acc = min(1, float(len(matchingAns))/3)
                gtAcc.append(acc)
            quesType    = gts[quesId]['question_type']
            ansType     = gts[quesId]['answer_type']
            avgGTAcc = float(sum(gtAcc))/len(gtAcc)
            accQA.append(avgGTAcc)
            if quesType not in accQuesType:
                accQuesType[quesType] = []
            accQuesType[quesType].append(avgGTAcc)
            if ansType not in accAnsType:
                accAnsType[ansType] = []
            if answerClass not in accAnswerClass:
                accAnswerClass[answerClass] = []
            accAnswerClass[answerClass].append(avgGTAcc)
            accAnsType[ansType].append(avgGTAcc)
            self.setEvalQA(quesId, avgGTAcc)
            self.setEvalQuesType(quesId, quesType, avgGTAcc)
            self.setEvalAnsType(quesId, ansType, avgGTAcc)
            if step%100 == 0:
                self.updateProgress(step/float(len(quesIds)))
            step = step + 1
        self.setAccuracy(accQA, accAnswerClass, accQuesType, accAnsType)
        print "Done computing accuracy"

    def setAccuracy(self, accQA, accAnswerClass, accQuesType, accAnsType):
        self.accuracy['overall']  = round(100*float(sum(accQA))/len(accQA), self.n)
        self.accuracy['classNormalizedOverall'] = \
                round(100*float(sum([sum(x)/len(x) for x in accAnswerClass.values()]))/len(accAnswerClass), self.n)
        self.accuracy['perAnswerClass'] = \
                {answerClass: round(100*float(sum(accAnswerClass[answerClass]))/len(accAnswerClass[answerClass]), self.n) for answerClass in accAnswerClass}
        self.accuracy['perQuestionType'] = \
                {quesType: round(100*float(sum(accQuesType[quesType]))/len(accQuesType[quesType]), self.n) for quesType in accQuesType}
        self.accuracy['perAnswerType']   = \
                {ansType:  round(100*float(sum(accAnsType[ansType]))/len(accAnsType[ansType]), self.n) for ansType in accAnsType}
 
