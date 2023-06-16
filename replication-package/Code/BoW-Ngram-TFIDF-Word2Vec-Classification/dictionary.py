import load_data
import _pickle as pkl
import gzip
import numpy as np
import pandas as pd
import operator
from sklearn.feature_extraction.text import CountVectorizer


def vectorizer(train, gram):
    print('Count Vectorizer ...')
    vectorizer = CountVectorizer(ngram_range = gram, strip_accents = 'ascii')
    X = vectorizer.fit_transform(train)
    word_column = vectorizer.get_feature_names()

    return word_column, X.toarray()


def create_dict(word, array):
    dictionary = {}
    count_word = [sum(x) for x in zip(*array)]
    print('Creating dictionary ...')
    for index in range(len(word)):
        if word[index] in dictionary:
            dictionary[word[index]] += count_word[index]
        else:
            dictionary[word[index]] = count_word[index]
    persistent_dict = sort_highest_frequency(dictionary, 100)
    return persistent_dict


def sort_highest_frequency(dict, length):
    print('Sorting top', length, 'frequency ...')
    print('All dictionary length: ', len(dict))
    sorted_d = sorted(dict.items(), key = operator.itemgetter(1), reverse = True)
    sorted_d = np.array(sorted_d)

    top_f = pd.DataFrame(sorted_d[0:length, 0])

    sort_word = pd.DataFrame(top_f.sort_values(0))
    sort_word = sort_word.reset_index(drop = True)

    persistent_dict = {}
    for index, word in enumerate(sort_word.values):
        persistent_dict[word[0]] = index
    return persistent_dict


def save_dict(project, preprocess, dictionary):
    f = gzip.open('Data/' + project + '_' + preprocess + '.dict.pkl.gz', 'wb')
    pkl.dump(dictionary, f, 2)
    f.close()
    print('Successfully creating dictionary!')


def startloop(dictionary, train, length, ngram):
    start = 0
    num = 5000
    for index in range(num, length, num):
        if index / num == length / num:
            sub_train = train[start:index]
            word, array = vectorizer(sub_train, ngram)
            create_dict(dictionary, word, array)
            sub_train = train[index:length]
            word, array = vectorizer(sub_train, ngram)
            create_dict(dictionary, word, array)
        else:
            sub_train = train[start:index]
            word, array = vectorizer(sub_train, ngram)
            create_dict(dictionary, word, array)
            start += num


def build_dict(project, preprocess, ngram, texttovec):
    data = load_data.dataset(project, preprocess)
    train_index, valid_index, test_index = load_data.experimental_sets(project)
    train = data['summary'].map(str) + ' ' + data['description']
    train = train[:train_index].apply(str).values
    word, array = vectorizer(train, ngram)
    dictionary = create_dict(word, array)
    preprocess += '_' + texttovec
    save_dict(project, preprocess, dictionary)


def build_dict_with_code(project, ngram):
    data = load_data.dataset(project)
    train_index, valid_index, test_index = load_data.experimental_sets(project)
    sdtrain = data['summary'].map(str) + ' ' + data['description']
    sdtrain = sdtrain.values
    code = data['code']
    codetrain = code[:train_index].values

    dictsd = {}
    dictcode = {}
    startloop(dictsd, sdtrain, train_index, ngram)
    startloop(dictcode, codetrain, train_index, ngram)
    dictsd = sort_highest_frequency(dictsd, 1000)
    dictcode = sort_highest_frequency(dictcode, 1000)

    save_dict('sd', project, dictsd)
    save_dict('code', project, dictcode)


def build_dict_ngram_idf(project, preprocess, length):
    path = 'Data/' + project + '_' + preprocess + '_ngweight_output.txt'
    print('Load ' + project + '_' + preprocess + '_ngweight_output.txt')
    data = pd.read_csv(path, delimiter = '\t', names = ['id', 'len', 'gtf', 'df', 'sdf', 'term'])

    data = data.sort_values(by = 'gtf', ascending = False)
    data = data.reset_index(drop = True)

    dict = data[:length].sort_values(by = 'term')
    dict = dict.reset_index(drop = True)

    dictionary = {}
    for index, term in enumerate(dict['term']):
        dictionary[term.strip()] = index
    print('Dictionary length: ', len(dictionary))
    preprocess += '_n-gram idf'
    save_dict(project, preprocess, dictionary)
