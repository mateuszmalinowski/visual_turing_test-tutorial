"""
DAQUAR dataset provider.

Ashkan Mokarian [ashkan@mpi-inf.mpg.de]
Mateusz Malinowski [mmalinow@mpi-inf.mpg.de]
"""

import copy
import os
import re
import json
import numpy as np

from read_write import file2list
from toolz import frequencies

from scipy.misc import imread

def daquar_qa_triples(
        path=None, 
        train_or_test='train', 
        keep_top_qa_pairs=0,
        **kwargs):
    """
    DAQUAR question answer pairs.

    In:
        path - path to DAQUAR root folder, if None then default path is chosen
            by default None
        train_or_test - switch between train and test set;
            value belongs to \{'train', 'val', 'test'\} 
            by default 'train'
        keep_top_qa_pairs - filter out question-answer pairs to the
            keep_top_qa_pairs if positive; by default 0

    Out:
        x - textual questions
        y - textual answers
        img_name - names of the images
        img_ind - image indices that correspond to x
        question_id - empty list as it is unused in DAQUAR
        end_of_question - end of question token
        end_of_answer - end of answer token
        answer_words_delimiter - delimiter for multiple word answers
    """
    if path is None:
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(curr_dir, '..', '..', 'data', 'daquar')

    if train_or_test == 'val':
        # we don't have a well established split
        train_or_test = 'train'

    xy_list = file2list(
            os.path.join(path,'qa.894.raw.'+train_or_test+'.format_triple'))

    # create a dictionary of allowed qa pairs
    all_answers = xy_list[1::3]
    freq = frequencies(all_answers)
    if keep_top_qa_pairs <= 0:
        most_frequent_answers = sorted(
                freq.items(), key=lambda x:x[1], reverse=True)
    else:
        most_frequent_answers = sorted(
                freq.items(), key=lambda x:x[1], reverse=True)[:keep_top_qa_pairs]
    allowed_answers_dict = dict(most_frequent_answers)
    #

    x_list = []
    y_list = []
    img_name_list = []
    img_ind_list = []
    for x, y, image_name in zip(xy_list[::3], xy_list[1::3], xy_list[2::3]):
        if y in allowed_answers_dict:
            x_list.append(x)
            y_list.append(y)
            img_name_list.append(image_name)
            img_num = re.search('(?<=image)[0-9]+', image_name).group(0)
            img_ind_list.append(int(img_num)-1)

    return {'x':x_list, 
            'y':y_list, 
            'img_name':img_name_list, 
            'img_ind': img_ind_list, 
            'question_id': [],
            'end_of_question':'?', 
            'end_of_answer':'',
            'answer_words_delimiter':','}


def daquar_save_results(question_id_list, answer_list, path):
    raise NotImplementedError()


def vqa_save_results(question_id_list, answer_list, path):
    """
    Saves the answers on question_id_list in the VQA-like format.

    In:
        question_id_list - list of the question ids
        answer_list - list with the answers
        path - path where the file is saved
    """
    question_answer_pairs = []
    assert len(question_id_list) == len(answer_list), \
            'must be the same number of questions and answers'
    for q,a in zip(question_id_list, answer_list):
        question_answer_pairs.append({'question_id':q, 'answer':str(a)})
    with open(path,'w') as f:
        json.dump(question_answer_pairs, f)


def vqa_get_object(path=None, train_or_test='train', 
        dataset_type='mscoco', task_type='OpenEnded', 
        annotation_year='2014', question_year='2015'):
    """
    In:
        path - path to VQA root folder, if None then default path is chosen;
            by default None
        train_or_test - switch between train and test set;
            value belongs to \{'train', 'val', 'test', 'test_dev'\} 
            by default 'train'
        dataset_type - type of dataset, e.g. 'mscoco'
        task_type - type of the task, e.g. 'OpenEnded'
        annotation_year - annotation year
        question_year - question year

    Out:
        root_path - constructed root path
        anno_path - constructed path to annotations
        questions_path - constructed path to questions
        vqa_object - constructed VQA object
    """
     
    from vqaTools.vqa import VQA     
    if path == None:
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        root_path = os.path.join(curr_dir, '..', '..', 'data', 'vqa')
    else: 
        root_path = path
    
    train_or_test_questions = 'test-dev' if train_or_test == 'test_dev' \
            else train_or_test
    dataset_train_or_test = train_or_test + annotation_year
    question_train_or_test = train_or_test_questions + question_year

    if train_or_test == 'test_dev':
        anno_path = None
    else:
        anno_path = os.path.join(root_path, 
                'Annotations', '{0}_{1}_annotations.json'.format(
                    dataset_type, dataset_train_or_test))
    questions_path = os.path.join(root_path,
            'Questions', '{0}_{1}_{2}_questions.json'.format(
                task_type, dataset_type, question_train_or_test))
    vqa = VQA(anno_path, questions_path)
    return {'root_path':root_path,
            'anno_path':anno_path,
            'questions_path':questions_path,
            'vqa_object':vqa}


def vqa_general(path=None, train_or_test='train', dataset_type='mscoco', 
        task_type='OpenEnded', annotation_year='2014', question_year='2015', 
        image_name_template='COCO_2014_{0:0=12}', answer_mode='single_random',
        keep_top_qa_pairs=0):
    """
    VT-Vision-Lab VQA question answeir pairs. It is a general interface.
    In:
        path - path to VQA root folder, if None then default path is chosen;
            by default None
        train_or_test - switch between train and test set;
            value belongs to \{'train', 'val', 'test', 'test_dev'\} 
            by default 'train'
        dataset_type - type of dataset, e.g. 'mscoco'
        task_type - type of the task, e.g. 'OpenEnded'
        annotation_year - annotation year
        question_year - question year
        image_name_template - template for giving names to images
        answer_mode - possible answer modes:
            'single_random' - single answer, randomly chosen
            'single_confident' - single answer, randomly chosen among the confident;
                if there is no confident then randomly chosen (the same as single)
            'single_frequent' - the most frequent answer
            'all' - with one question all answers
            'all_repeat' - all answers by repeating the same question
            'all_repeat_confidentonly' - all answers that are confident (repeats the same question)
        keep_top_qa_pairs - filter out question-answer pairs to the
            keep_top_qa_pairs if positive; by default 0

    Out:
        x - textual questions
        y - textual answers
        img_name - names of the images
        img_ind - image indices that correspond to x
        question_id - list of question indices
        end_of_question - end of question token
        end_of_answer - end of answer token
        answer_words_delimiter - delimiter for multiple word answers
        anno_path - constructed path to annotations
        questions_path - constructed path to questions
    """

    def preprocess_question(q):
        q_tmp = q.strip().lower().encode('utf8')
        if q_tmp[-1] == '?' and q_tmp[-2] != ' ':
            # separate word token from the question mark
            q_tmp = q_tmp[:-1] + ' ?'
        # remove question mark
        if q_tmp[-1] == '?': q_tmp = q_tmp[:-1]
        return q_tmp
    #

    assert answer_mode in ['single_random', 'single_confident', 'single_frequent', 'all', 'all_repeat', 'all_repeat_confidentonly']
    assert task_type in ['OpenEnded', 'MultipleChoice'], \
            'The task is either ''OpenEnded'' of ''MultipleChoice'''
    assert dataset_type in ['mscoco', 'abstract_v002'], \
            'The type of dataset is eigher ''mscoco'' or ''abstract_v002'''

    vqa_dict = vqa_get_object(
            path=path, 
            train_or_test=train_or_test, 
            dataset_type=dataset_type, 
            task_type=task_type, 
            annotation_year=annotation_year, 
            question_year=question_year)
    vqa = vqa_dict['vqa_object']

    # questions can be filtered, e.g. by the question type
    ann_ids = vqa.getQuesIds()     
    anns = vqa.loadQA(ann_ids)
   
    # process annotations
    question_id_list = []
    image_name_list = []
    image_id_list = []
    x_list = []
    y_list = []

    # return only questions if there are no annotations
    if anns == []:
        for ques in vqa.questions['questions']:
            question = preprocess_question(ques['question'])
            x_list.append(question)
            question_id_list.append(ques['question_id'])
            image_id = ques['image_id']
            image_name = image_name_template.format(image_id)
            image_name_list.append(image_name)
            image_id_list.append(image_id)

    # create a dictionary of allowed qa pairs
    all_answers = [x['answer'] for anno in anns for x in anno['answers']]
    freq = frequencies(all_answers)
    if keep_top_qa_pairs <= 0:
        most_frequent_answers = sorted(
                freq.items(), key=lambda x:x[1], reverse=True)
    else:
        most_frequent_answers = sorted(
                freq.items(), key=lambda x:x[1], reverse=True)[:keep_top_qa_pairs]
    allowed_answers_dict = dict(most_frequent_answers)
    #

    for anno in anns:
        image_id = anno['image_id']
        image_name = image_name_template.format(image_id)
        question_id = anno['question_id']
        question = preprocess_question(vqa.qqa[question_id]['question'])
        assert image_id == vqa.qqa[question_id]['image_id'], \
                'image id of the question and answer are different'
        # randomizing the answers list
        randomized_answers = copy.deepcopy(anno['answers'])
        np.random.shuffle(randomized_answers)
        randomized_allowed_answers_list = \
                [x for x in randomized_answers if x['answer'] in allowed_answers_dict]
        if randomized_allowed_answers_list == []:
            continue
        #
        if answer_mode == 'single_random':
            answer = randomized_allowed_answers_list[0]['answer']
        elif answer_mode == 'single_confident':
            # if there is no confident answer, take a random one
            confidence_list = [x['answer_confidence'] \
                    for x in randomized_allowed_answers_list]
            yes_list = [j for j,x in enumerate(confidence_list) if x == 'yes'] 
            if yes_list == []:
                answer = randomized_allowed_answers_list[0]['answer']
            else:
                answer = randomized_allowed_answers_list[yes_list[0]]['answer']
        elif answer_mode == 'single_frequent':
            tmp = frequencies([x['answer'] for x in randomized_allowed_answers_list])
            answer = sorted(tmp.items(), key=lambda x: x[1], reverse=True)[0][0]
        elif answer_mode == 'all':
            raise NotImplementedError()
        elif answer_mode == 'all_repeat':
            answer_list_all_mode = []
            for answer in randomized_allowed_answers_list:
                answer_list_all_mode.append(answer['answer'].encode('utf8'))
        elif answer_mode == 'all_repeat_confidentonly':
            # like repeat but consider only confident answers
            confidence_list = [x['answer_confidence'] \
                    for x in randomized_allowed_answers_list]
            yes_list = [j for j,x in enumerate(confidence_list) if x == 'yes'] 
            if yes_list == []:
                # we keep only confident qa pairs
                continue
            answer_list_all_mode = []
            for answer_no, answer in enumerate(randomized_allowed_answers_list):
                if answer_no in yes_list:
                    answer_list_all_mode.append(answer['answer'].encode('utf8'))
        else:
            raise NotImplementedError()

        if 'single' in answer_mode:
            answer = answer.encode('utf8')
            x_list.append(question)
            y_list.append(answer)
            image_name_list.append(image_name)
            image_id_list.append(image_id)
            question_id_list.append(question_id)
        elif 'all' in answer_mode:
            num_answers_all_mode = len(answer_list_all_mode)
            x_list.extend([question]*num_answers_all_mode)
            image_name_list.extend([image_name]*num_answers_all_mode)
            image_id_list.extend([image_id]*num_answers_all_mode)
            question_id_list.extend([question_id]*num_answers_all_mode)
            y_list.extend(answer_list_all_mode)
        else:
            raise NotImplementedError()

    return {'x':x_list, 'y':y_list, 
            'img_name':image_name_list, 
            'img_ind': image_id_list, 
            'question_id': question_id_list,
            'end_of_question':'?', 
            'end_of_answer':'',
            'answer_words_delimiter':' ',
            'vqa_object':vqa,
            'questions_path':vqa_dict['questions_path'],
            'anno_path':vqa_dict['anno_path']}


def vqa_real_images_open_ended(
        path=None, 
        train_or_test='train', 
        keep_top_qa_pairs=0, 
        answer_mode='single',
        **kwargs):
    """
    VT-Vision-Lab VQA open-ended question answeir pairs.

    In:
        path - path to VQA root folder, if None then default path is chosen;
            by default None
        train_or_test - switch between train and test set;
            value belongs to \{'train', 'val', 'test', 'test_dev\} 
            by default 'train'
        answer_mode - possible answer modes:
            'single_random' - single answer, randomly chosen
            'single_confident' - single answer, randomly chosen among the confident;
                if there is no confident then randomly chosen (the same as single)
            'single_frequent' - the most frequent answer
            'all' - with one question all answers
            'all_repeat' - all answers by repeating the same question
            'all_repeat_confidentonly' - all answers that are confident (repeats the same question)
        keep_top_qa_pairs - filter out question-answer pairs to the
            keep_top_qa_pairs if positive; by default 0

    Out:
        x - textual questions
        y - textual answers
        img_name - names of the images
        img_ind - image indices that correspond to x
        question_id - list of question indices
        end_of_question - end of question token
        end_of_answer - end of answer token
        answer_words_delimiter - delimiter for multiple word answers
    """

    dataset_type = 'mscoco'
    annotation_year = '2014'
    question_year = '2015' if 'test' in train_or_test else '2014'
    task_type = 'OpenEnded'
    train_or_test_image = 'test' if 'test' in train_or_test else train_or_test
    image_name_template = 'COCO_' + train_or_test_image + question_year + '_{0:0=12}'

    return vqa_general(
            path=path,
            train_or_test=train_or_test,
            dataset_type=dataset_type,
            task_type=task_type,
            annotation_year=annotation_year,
            question_year=question_year,
            image_name_template=image_name_template,
            answer_mode=answer_mode,
            keep_top_qa_pairs=keep_top_qa_pairs)


###
# Non-dataset specific functions.
###
def is_image_file(x):
    return x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')


def global_visual_features(
        path, perception='googlenet', layer='pool5-7x7_s1',
        memory_time_steps=35, is_shuffle_memories=True, names_list=None):
    """
    Provides global visual features.

    In:
        path - the root path
        perception - the perception model; by default 'googlenet'
        layer - the layer in the model; by default 'pool5-7x7_s1'
        memory_time_steps - number of memories, everything outside is cut out; 
            valid only if visual features are 3d tensors; by default 35
        is_shuffle_memories - shuffle memories;
            it's more important when they must be truncated; by default True
        names_list - list of the image names, if None then all images are considered;
            only valid if data are stored as mappings from names into features;
            by default None
    """
    assert path is not None, 'Set up the path!'
    if is_shuffle_memories:
        print 'Shuffling memories ...'
    visual_features = np.load(os.path.join(
        path, perception, 'blobs.' + layer + '.npy'))
    if visual_features.shape == ():
        visual_features = visual_features.item()

    if names_list is None or names_list==[]:
        return visual_features
    else:
        # either 2D or 3D tensor
        tmp_feats = visual_features[visual_features.keys()[0]]
        if layer.endswith('index'):
            visual_features_subset = np.zeros(
                    (len(names_list), memory_time_steps), dtype=int)
            is_memories = True
        elif tmp_feats.ndim == 1:
            visual_features_subset = np.zeros(
                    (len(names_list), tmp_feats.shape[0]))
            is_memories = False
        elif tmp_feats.ndim == 2:
            # matrix has dimensions #images x #time_steps x #features
            visual_features_subset = np.zeros(
                    (len(names_list), memory_time_steps, tmp_feats.shape[-1]))
            is_memories = True
        else:
            raise NotImplementedError()
        skipped_image_names = set() 
        for k, name_now in enumerate(names_list):
            if name_now not in visual_features:
                # keep going if image doesn't exist in features
                skipped_image_names.add(name_now)
                continue
            if is_memories:
                visual_features_now = visual_features[name_now]
                number_memories = visual_features_now.shape[0]
                if is_shuffle_memories:
                    shuffled_memory_indices = \
                            np.arange(visual_features_now.shape[0])
                    np.random.shuffle(shuffled_memory_indices)
                    visual_features_now = \
                            visual_features_now[shuffled_memory_indices]
                if layer.endswith('index'):
                    # we add one because we want to mask-out zeroes
                    visual_features_subset[k,-number_memories:] = \
                            np.squeeze(visual_features_now[:memory_time_steps]+1)
                else:
                    visual_features_subset[k,-number_memories:,:] = \
                            visual_features_now[:memory_time_steps,:]
            else:
                visual_features_subset[k,:] = visual_features[name_now]
        print('Skipped images {0} of them:'.format(len(skipped_image_names)))
        for name_now in skipped_image_names:
            print(name_now)
        return visual_features_subset


def get_global_perception(
        task='daquar', train_or_test='train', extractor_fun=global_visual_features, 
        path=None, perception='googlenet', layer='pool5-7x7_s1', names_list=None):
    """
    Provides global visual features.

    In:
        task - the challenge; by default 'daquar'
        train_or_test - training, validation, or test set; by default train
        extractor_fun - function for extraction; 
            by default global_visual_features
        path - the root path, if None then default path is taken; 
            by default None
        perception - the perception model; by default 'googlenet'
        layer - the layer in the model; by default 'pool5-7x7_s1'
        names_list - list of the image names, if None then all images are considered;
            only valid if data are stored as mappings from names into features;
            by default None
    """
    if path is None:
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        root_path = os.path.join(curr_dir, '..', '..', 'data')
        if task=='daquar':
            task_path = os.path.join(root_path, 'daquar', 'visual_features')
        elif task == 'vqa':
            if train_or_test == 'train': vqa_train_or_test='train2014'
            elif train_or_test == 'val': vqa_train_or_test='val2014'
            elif 'test' in train_or_test: vqa_train_or_test='test2015'
            else: raise NotImplementedError()
            task_path = os.path.join(root_path, 'vqa',  'visual_features', vqa_train_or_test)
    else:
        task_path = path

    return extractor_fun(
            path=task_path, 
            perception=perception, 
            layer=layer, 
            names_list=names_list)


# Selector
###
select = {
        'daquar-triples': {
            'text':daquar_qa_triples, 
            'perception':lambda train_or_test, names_list, 
                    parts_extractor, max_parts, perception, 
                    layer, second_layer:
                get_global_perception(
                    task='daquar', 
                    train_or_test=train_or_test, 
                    names_list=names_list,
                    extractor_fun=global_visual_features,
                    perception=perception,
                    layer=layer), 
           'save_predictions': daquar_save_results
            },
        'vqa-real_images-open_ended': {
            'text':vqa_real_images_open_ended,
            'perception':lambda train_or_test, names_list, 
                    parts_extractor, max_parts, perception, 
                    layer, second_layer: 
                get_global_perception(
                    task='vqa', 
                    train_or_test=train_or_test, 
                    names_list=names_list,
                    extractor_fun=global_visual_features,
                    perception=perception,
                    layer=layer),
            'visual_parameters':lambda train_or_test, perception, params: 
                get_global_perception(
                    task='vqa', train_or_test=train_or_test, 
                    extractor_fun=global_visual_parameters,
                    perception=perception,
                    params=params),
            'save_predictions': vqa_save_results
            },
        }

