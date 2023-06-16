import pandas as pd
import numpy as np
import load_data


def prepare_y(actual, estimate, k):
    y_true = np.argwhere(actual)
    y_pred = estimate.argsort()[-k:]

    return y_true, y_pred


def write_file(score, metric, project, preprocess, model, texttovec):
    if preprocess == '':
        f = open('output/' + project + '_' + metric + '_' + model + '.txt', 'w')
    else:
        f = open('output/' + project + '_' + preprocess + '_' + texttovec + '_' + metric + '_' + model + '.txt', 'w')
    f.write(str(score))
    f.close()
    print('Writing ' + metric + ' successfully')


def write_evaluate_file(project, preprocess, method, model, list_result, texttovec):
    file = pd.DataFrame(list_result).T
    if preprocess == '':
        file.to_csv('output/' + project + '_' + method + '@k_' + model + '.txt', header = False, index = False)
    else:
        file.to_csv('output/' + project + '_' + preprocess + '_' + texttovec + '_' + method + '@k_' + model + '.txt',
                    header = False, index = False)
    print('Writing ' + method + '@k successfully')


def precision_at_k(project, preprocess, model, texttovec):
    method = 'precision'
    actual, estimate = load_data.actual_and_estimate_data(project, preprocess, model, texttovec)
    precision_at_k = []
    component_num = len(actual.T)

    for k in range(component_num):
        print((k + 1))
        sum_precision = 0.0
        for j in range(len(actual)):
            y_true, y_pred = prepare_y(actual[j], estimate[j], (k + 1))
            relevant = (y_true == y_pred).sum()
            total_recommended_item = k + 1
            sum_precision = sum_precision + (float(relevant) / total_recommended_item)

        precision_at_k.append('%.4f' % (sum_precision / len(actual)))
        print('Precision@K: {:.4f}'.format(sum_precision / len(actual)))

    write_evaluate_file(project, preprocess, method, model, precision_at_k, texttovec)


def recall_at_k(project, preprocess, model, texttovec):
    method = 'recall'
    actual, estimate = load_data.actual_and_estimate_data(project, preprocess, model, texttovec)
    recall_at_k = []
    if 'tuning' in model:
        component_num = 100
    else:
        component_num = len(actual.T)

    for k in range(component_num):
        print((k + 1))
        sum_recall = 0.0
        for j in range(len(actual)):
            y_true, y_pred = prepare_y(actual[j], estimate[j], (k + 1))
            relevant = (y_true == y_pred).sum()
            total_relevant = len(y_true)

            print(relevant)
            print(total_relevant)
            print("...")
            sum_recall = sum_recall + (float(relevant) / total_relevant)
            
        recall_at_k.append('%.4f' % (sum_recall / len(actual)))
        print('Recall@K: {:.4f}'.format(sum_recall / len(actual)))

    write_evaluate_file(project, preprocess, method, model, recall_at_k, texttovec)


def map(project, preprocess, model, texttovec):
    actual, estimate = load_data.actual_and_estimate_data(project, preprocess, model, texttovec)
    m = len(actual)
    sum = 0.0
    for j in range(len(actual)):
        y_true = np.argwhere(actual[j])
        tp = len(y_true)
        psum = 0.0
        for k in range(1, tp + 1):
            y_pred = estimate[j].argsort()[-k:]
            intersect = (y_true == y_pred).sum()
            prec = intersect / float(k)
            psum += prec
        ap = psum / float(tp)
        sum += ap
    mean_avg_prec = sum / float(m)
    print('Project: {}, Model: {}'.format(project, model))
    print('MAP: {:f} over {:.4f} issues'.format(mean_avg_prec, m))
    write_file(mean_avg_prec, 'MAP', project, preprocess, model, texttovec)


def mrr(project, preprocess, model, texttovec):
    actual, estimate = load_data.actual_and_estimate_data(project, preprocess, model, texttovec)
    sum_rr = 0.0
    m = len(actual)
    for j in range(len(actual)):
        y_true = np.argwhere(actual[j])
        y_pred = estimate[j].argsort()[::-1][:]
        ranks = []
        for idx, i in enumerate(y_pred):
            if i in y_true:
                ranks.append(idx + 1)
        if len(ranks) > 0:
            first = ranks[0]
            rr = 1 / float(first)
            sum_rr += rr
    mrr_score = sum_rr / float(m)
    print('Project: {}, Model: {}'.format(project, model))
    print('MRR: {:f} over {:.4f} issues'.format(mrr_score, m))
    write_file(mrr_score, 'MRR', project, preprocess, model, texttovec)


def wilcoxon_precision(project, preprocess, model, texttovec, num):
    actual, estimate = load_data.actual_and_estimate_data(project, preprocess, model, texttovec)
    pre_list = []
    print('Precision@', num)
    for j in range(len(actual)):
        y_true, y_pred = prepare_y(actual[j], estimate[j], num)
        relevant = (y_true == y_pred).sum()
        pre_list.append(float(relevant) / num)
    pre_list = np.vstack(pre_list)
    df = pd.DataFrame(data = pre_list)
    if preprocess == '' or texttovec == '':
        df.to_csv('output/wilcoxon/' + project + '_precision@' + str(num) + '_' + model + '.csv', 'w', index = False)
    else:
        df.to_csv('output/wilcoxon/' + project + '_' + preprocess + '_' + texttovec + '_precision@' + str(
            num) + '_' + model + '.csv', 'w', index = False)
    print('Writing wilconxon list Precision@' + str(num) + ' successfully')


def wilcoxon_recall(project, preprocess, model, texttovec, num):
    actual, estimate = load_data.actual_and_estimate_data(project, preprocess, model, texttovec)
    re_list = []
    print('Recall@', num)
    for j in range(len(actual)):
        y_true, y_pred = prepare_y(actual[j], estimate[j], num)
        relevant = (y_true == y_pred).sum()
        total_relevant = len(y_true)
        re_list.append(float(relevant) / total_relevant)

    re_list = np.vstack(re_list)
    df = pd.DataFrame(data = re_list)
    if preprocess == '' or texttovec == '':
        df.to_csv('output/wilcoxon/' + project + '_recall@' + str(num) + '_' + model + '.csv', 'w', index = False)
    else:
        df.to_csv('output/wilcoxon/' + project + '_' + preprocess + '_' + texttovec + '_recall@' + str(
            num) + '_' + model + '.csv', 'w', index = False)
    print('Writing wilconxon list Recall@' + str(num) + ' successfully')


def wilcoxon_map(project, preprocess, model, texttovec):
    actual, estimate = load_data.actual_and_estimate_data(project, preprocess, model, texttovec)
    map_list = []
    print('MAP')
    for j in range(len(actual)):
        y_true = np.argwhere(actual[j])
        tp = len(y_true)
        psum = 0.0
        for k in range(1, tp + 1):
            y_pred = estimate[j].argsort()[-k:]
            intersect = (y_true[-k:] == y_pred).sum()
            prec = intersect / float(k)
            psum += prec
        ap = psum / float(tp)
        map_list.append(ap)

    map_list = np.vstack(map_list)
    df = pd.DataFrame(data = map_list)
    if preprocess == '' or texttovec == '':
        df.to_csv('output/wilcoxon/' + project + '_map_' + model + '.csv', 'w', index = False)
    else:
        df.to_csv('output/wilcoxon/' + project + '_' + preprocess + '_' + texttovec + '_map_' + model + '.csv', 'w',
                  index = False)
    print('Writing wilconxon list map successfully')


def wilcoxon_mrr(project, preprocess, model, texttovec):
    actual, estimate = load_data.actual_and_estimate_data(project, preprocess, model, texttovec)
    mrr_list = []
    print('MRR')
    for j in range(len(actual)):
        y_true = np.argwhere(actual[j])
        y_pred = estimate[j].argsort()[::-1][:]
        ranks = []
        for idx, i in enumerate(y_pred):
            if i in y_true:
                ranks.append(idx + 1)
        if len(ranks) > 0:
            first = ranks[0]
            rr = 1 / float(first)
        mrr_list.append(rr)
    mrr_list = np.vstack(mrr_list)
    df = pd.DataFrame(data = mrr_list)
    if preprocess == '' or texttovec == '':
        df.to_csv('output/wilcoxon/' + project + '_mrr_' + model + '.csv', 'w', index = False)
    else:
        df.to_csv('output/wilcoxon/' + project + '_' + preprocess + '_' + texttovec + '_mrr_' + model + '.csv', 'w',
                  index = False)
    print('Writing wilconxon list mrr successfully')
