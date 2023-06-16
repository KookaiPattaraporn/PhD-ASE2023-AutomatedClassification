import re
import pandas as pd
import math
from nltk.corpus import stopwords


def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]', ' ', text)


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def remove_special_char(sentence):
    return re.sub('[^A-Za-z0-9]+', ' ', sentence)


def loop_sw(column, array):
    for i, sentence in enumerate(array):
        print('Index-', i)
        sentence_clean = ''
        print('Lower case')
        try:
            sentence = sentence.lower()
        except:
            sentence = "Nan"
        if is_ascii(sentence) == False:
            print('Remove ASCII')
            sentence = remove_non_ascii(sentence)
        print('Remove special character')
        sentence = remove_special_char(sentence)
        words = sentence.split(' ')
        print(words)
        for word in words:
            if word not in stop_words:
                sentence_clean += word + ' '
            else:
                print('Stopword: ', word)

        print('Remove spacebar')
        if sentence_clean == '':
            data[column][i] = 'none'
        else:
            data[column][i] = sentence_clean.strip()
        print(data[column][i])

def loop_ascii(column, array):
    for i, sentence in enumerate(array):
        print('Index-', i)
        print('Lower case')
        sentence = sentence.lower()
        if is_ascii(sentence) == False:
            print('Remove ASCII')
            sentence = remove_non_ascii(sentence)
        print('Remove special character')
        sentence = remove_special_char(sentence)
        data[column][i] = sentence.strip()
        print(data[column][i])

stop_words = set(stopwords.words('english'))

dataset_list = ['Moodle_reg_level']
for project in dataset_list:
    data = pd.read_csv('Data/' + project + '.csv', na_values=" Nan")
    
    title = data['summary'].values
    description = data['description'].values

    loop_sw('summary', title)
    loop_sw('description', description)

    data.to_csv('Data/' + project + '(remove stopwords).csv', index = False)

    print('Finish')

def preprocess(project):
    data = pd.read_csv('Data/' + project + '.csv', na_values=" Nan")

    title = data['summary'].values
    description = data['description'].values

    loop_sw('summary', title)
    loop_sw('description', description)

    data.to_csv('Data/' + project + '(remove stopwords).csv', index = False)

    print('Finish')
