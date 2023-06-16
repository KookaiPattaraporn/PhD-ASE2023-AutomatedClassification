import baseline
import dictionary
import experimental_sets
import evaluate
import pre_process
import prepare_data
import rf
import datetime

import spacy_preprocess
import sys


dataset_list = ['Chrome_reg_level','Moodle_reg_level']


preprocess_list = ['ascii', 'remove stopwords', 'lemmatization', 'all']
ttv_list = ['BOW', 'n-gram idf', 'tfidf']
model_list = ['baselines', 'RF']

#Change project data, text_to_vec and model here
project = 'Moodle_reg_level'
preprocess = 'all'
text_to_vec = 'n-gram idf'
model = 'RF'

if project in dataset_list and preprocess in preprocess_list and text_to_vec in ttv_list and model in model_list:
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    experimental_sets.create_file(project)
    prepare_data.create_list_labels(project)

    if model == 'baselines':
        baseline.predict_freq(project)
        baseline.predict_random_issue(project)
        evaluate.recall_at_k(project, '', 'frequency', '')
        evaluate.precision_at_k(project, '', 'frequency', '')
        evaluate.map(project, '', 'frequency', '')
        evaluate.mrr(project, '', 'frequency', '')
        evaluate.recall_at_k(project, '', 'random_issue', '')
        evaluate.precision_at_k(project, '', 'random_issue', '')
        evaluate.map(project, '', 'random_issue', '')
        evaluate.mrr(project, '', 'random_issue', '')

    elif model == 'RF':
        pre_process.preprocess(project)
        spacy_preprocess.all(project)
        dictionary.build_dict(project,'all',(1, 10),text_to_vec)
        # prepare_data.create_pkl(project,'all',(1, 1),text_to_vec)
        prepare_data.create_pkl_tfidf(project,'all',(1, 10),text_to_vec)
        rf.train_rf(project, preprocess, model, text_to_vec)
        rf.create_estimate(project, preprocess, model, text_to_vec)
        evaluate.recall_at_k(project, preprocess, model, text_to_vec)
        evaluate.precision_at_k(project, preprocess, model, text_to_vec)
        evaluate.map(project, preprocess, model, text_to_vec)
        evaluate.mrr(project, preprocess, model, text_to_vec)

else:
    print('Invalid input')
