import pandas as pd
import numpy as np
import load_data


def predict_freq(project):
    data = load_data.dataset(project, '')
    train_index, valid_index, test_index = load_data.experimental_sets(project)

    labels = data.iloc[:train_index + valid_index, 4:]
    component_count = [sum(x) for x in zip(*labels.values)]

    df = pd.DataFrame(np.array(component_count).reshape(-1, len(component_count)))
    df = pd.concat([df] * test_index)
    df.to_csv('Data/' + project + '_frequency_estimate.csv', index = False, header = False)

    print('Writing estimate successfully')

    component_index = np.array(component_count).argsort()[::-1][:]

    recommended_component = []

    for x in component_index:
        recommended_component.append(labels.columns[x])

    df_recommended_component = pd.DataFrame(recommended_component).T
    df_recommended_component.to_csv('output/' + project + '_result_frequency.txt', header = False, index = False)

    print('Writing frequency result successfully')

    return recommended_component


def predict_random_issue(project):
    data = load_data.dataset(project, '')
    train_index, valid_index, test_index = load_data.experimental_sets(project)
    train = data.iloc[:train_index + valid_index, 4:]
    random = train.sample(n = test_index, random_state = 1)
    random.to_csv('Data/' + project + '_random_issue_estimate.csv', header = False, index = False)
    print('Writing random issue result successfully')
