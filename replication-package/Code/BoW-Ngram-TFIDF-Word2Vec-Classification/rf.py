import pandas as pd
import load_data
import classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
import datetime


def train_rf(project, preprocess, model, text_to_vec):
    train_x, _, _, train_y, _, _ = load_data.prepare_data(project, preprocess, text_to_vec)
    clf = RandomForestClassifier(max_features = 'auto', n_estimators = 150, max_depth = None, min_samples_split = 2,
                                 random_state = 1, n_jobs = -1)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print('Training model ...')
    clf.fit(train_x, train_y)
    print('Successfully train model')
    classifier.save(project, preprocess, text_to_vec, model, clf)


def create_estimate(project, preprocess, model, text_to_vec):
    _, _, test_x, _, _, _ = load_data.prepare_data(project, preprocess, text_to_vec)

    clf = load_data.model(project, preprocess, text_to_vec, model)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(len(pd.DataFrame(test_x).T))
    print('Predicting probability ...')
    prediction = clf.predict_proba(test_x)
    print('Successful predicting probability')
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    y_pred = []
    print('Building estimation array ...')
    for component_index in range(len(prediction)):
        issue_prob = []
        for issue in range(len(prediction[component_index])):
            if len(prediction[component_index][issue]) == 1:
                issue_prob.append(0)
            else:
                issue_prob.append(prediction[component_index][issue][1])
        y_pred.append(issue_prob)
    print('Successful building estimation array')
    print('Writing ' + model + ' model estimation file ...')
    df = pd.DataFrame(data = y_pred)
    df.T.to_csv('Data/' + project + '_' + preprocess + '_' + text_to_vec + '_' + model + '_estimate.csv', index = False,
                header = False)
    print('Successful writing estimation file')


def train_OvR(project, model):
    train_x, valid_x, test_x, train_y, valid_y, test_y = load_data.prepare_data(project)
    _, _, test_num = load_data.experimental_sets(project)
    clf = OneVsRestClassifier(
        RandomForestClassifier(max_features = 'auto', n_estimators = 100, max_depth = None, min_samples_split = 2,
                               random_state = 1))
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print('Training model: ')
    clf.fit(train_x, train_y)
    print('Predict probability')
    prediction = clf.predict_proba(test_x)

    print('Succesfully predict probability')
    print('Building estimation file ...')
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    data = pd.DataFrame(prediction)
    data.to_csv('Data/' + project + '_' + model + '_estimate.csv', index = None, header = None)
    print('Building estimation file successfully')
    classifier.save(project, model, clf)
