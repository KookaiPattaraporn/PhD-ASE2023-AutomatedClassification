# ***********python3*********************
import pandas as pd
import re
import datetime
import spacy


def dataset(project):
    dataframe = pd.read_csv('Data/' + project + '.csv')
    print('Loading ' + project + '.csv')
    return dataframe


def remove_special_char(sentence):
    return str(re.sub('[^A-Za-z0-9]+', ' ', sentence).strip())


def lemmatization(project):
    print('Project name: ', project)
    data = dataset(project)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    title = data['summary'].values
    description = data['description'].values

    nlp = spacy.load('en_core_web_sm')

    for index, (title_sentence, des_sentence) in enumerate(zip(title, description)):
        print('Issue: ', index)
        if title_sentence == '':
            title_sentence = 'none'
        if des_sentence == '':
            des_sentence = 'none'
        title_doc = nlp(title_sentence)
        des_doc = nlp(des_sentence)
        new_title = ''
        new_des = ''
        for t_token in title_doc:
            if t_token.is_stop is not True:
                new_title += t_token.lemma_ + ' '
        for des_token in des_doc:
            if des_token.is_stop is not True:
                new_des += des_token.lemma_ + ' '

        data['summary'][index] = remove_special_char(new_title.lower())
        data['description'][index] = remove_special_char(new_des.lower())
        print('summary: ', data['summary'][index])
        print('Description: ', data['description'][index])

    data.to_csv('Data/' + project + '(lemmatization).csv', index = False)
    print('Finish writing new dataset: ', project + '(lemmatization).csv')
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def all(project):
    print('Project name: ', project + '(remove stopwords)')
    data = dataset(project + '(remove stopwords)')
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    title = data['summary'].values
    description = data['description'].values

    nlp = spacy.load('en_core_web_sm')

    for index, (title_sentence, des_sentence) in enumerate(zip(title, description)):
        print('Issue: ', index)
        if title_sentence == '':
            title_sentence = 'none'
        if des_sentence == '':
            des_sentence = 'none'
        title_doc = nlp(title_sentence)
        des_doc = nlp(des_sentence)
        new_title = ''
        new_des = ''
        for t_token in title_doc:
            if t_token.is_stop is not True:
                new_title += t_token.lemma_ + ' '
        for des_token in des_doc:
            if des_token.is_stop is not True:
                new_des += des_token.lemma_ + ' '

        data['summary'][index] = remove_special_char(new_title.lower())
        data['description'][index] = remove_special_char(new_des.lower())
        print('summary: ', data['summary'][index])
        print('Description: ', data['description'][index])

    data.to_csv('Data/' + project + '(all).csv', index = False)
    print('Finish writing new dataset: ', project + '(all).csv')
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
