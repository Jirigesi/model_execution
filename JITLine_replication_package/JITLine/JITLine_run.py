# import pickle
from JITLine_replication_package.JITLine.my_util import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, precision_recall_fscore_support, classification_report, auc

from imblearn.over_sampling import SMOTE

import numpy as np
from scipy.optimize import differential_evolution
import pandas as pd
import time, pickle, math, warnings, os

warnings.filterwarnings('ignore')
projects = ['openstack','qt']
sampling_methods = 'DE_SMOTE_min_df_3'
remove_python_common_tokens = True

def get_combined_df(code_commit, commit_id, label, metrics_df, count_vect):
    code_df = pd.DataFrame()
    code_df['commit_id'] = commit_id
    code_df['code'] = code_commit
    code_df['label'] = label

    code_df = code_df.sort_values(by='commit_id')

    metrics_df = metrics_df.sort_values(by='commit_id')
    metrics_df = metrics_df.drop('commit_id',axis=1)

    code_change_arr = count_vect.transform(code_df['code']).astype(np.int16).toarray()
    metrics_df_arr = metrics_df.to_numpy(dtype=np.float32)

    final_features = np.concatenate((code_change_arr,metrics_df_arr),axis=1)

    return final_features, list(code_df['commit_id']), list(code_df['label'])

def objective_func(k, train_feature, train_label, valid_feature, valid_label):
    smote = SMOTE(random_state=42, k_neighbors= int(np.round(k)), n_jobs=32)
    train_feature_res, train_label_res = smote.fit_resample(train_feature, train_label)

    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(train_feature_res, train_label_res)

    prob = clf.predict_proba(valid_feature)[:,1]
    auc = roc_auc_score(valid_label, prob)

    return -auc

def run_experiment(cur_proj):
    data_path = './data/'
    model_path = './final_model/'

    train_code, train_commit, train_label = prepare_data(cur_proj, mode='train',
                                                                  remove_python_common_tokens=remove_python_common_tokens)
    test_code, test_commit, test_label = prepare_data(cur_proj, mode='test',
                                                              remove_python_common_tokens=remove_python_common_tokens)

    commit_metrics = load_change_metrics_df(cur_proj)
    train_commit_metrics = commit_metrics[commit_metrics['commit_id'].isin(train_commit)]
    test_commit_metrics = commit_metrics[commit_metrics['commit_id'].isin(test_commit)]

    count_vect = CountVectorizer(min_df=3, ngram_range=(1,1))
    count_vect.fit(train_code)

    train_feature, train_commit_id, new_train_label = get_combined_df(train_code, train_commit, train_label, train_commit_metrics,count_vect)
    test_feature, test_commit_id, new_test_label = get_combined_df(test_code, test_commit, test_label, test_commit_metrics,count_vect)

    percent_80 = int(len(new_train_label)*0.8)

    final_train_feature = train_feature[:percent_80]
    final_train_commit_id = train_commit_id[:percent_80]
    final_new_train_label = new_train_label[:percent_80]

    valid_feature = train_feature[percent_80:]
    valid_commit_id = train_commit_id[percent_80:]
    valid_label = new_train_label[percent_80:]

    print('load data of',cur_proj, 'finish')

    bounds = [(1,20)]
    result = differential_evolution(objective_func, bounds, args=(final_train_feature, final_new_train_label,
                                                                  valid_feature, valid_label),
                                   popsize=10, mutation=0.7, recombination=0.3,seed=0)

    

    smote = SMOTE(random_state=42, n_jobs=32, k_neighbors=int(np.round(result.x)))
    train_feature_res, train_label_res = smote.fit_resample(final_train_feature, final_new_train_label)

    start_time = time.time()

    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf_name = 'RF'
    trained_clf, pred_df = train_eval_model(clf, train_feature_res, train_label_res,
                                       test_feature, new_test_label)
    
    print("--- Build model %s seconds ---" % (time.time() - start_time))

    pred_df['test_commit'] = test_commit_id
    pred_df.to_csv(data_path+cur_proj+'_'+clf_name+'_'+sampling_methods+'_prediction_result.csv')

    model_path = model_path+cur_proj+'_'+clf_name+'_'+sampling_methods+'.pkl'
    pickle.dump(trained_clf, open(model_path, 'wb'))

    print('finished',cur_proj)
    print('-'*100)

    k_of_smote = result.x
    best_AUC_of_obj_func = result.fun

    return k_of_smote, best_AUC_of_obj_func



def get_recall_at_k_percent_effort(percent_effort, result_df_arg, real_buggy_commits):
    cum_LOC_k_percent = (percent_effort/100)*result_df_arg.iloc[-1]['cum_LOC']
    buggy_line_k_percent =  result_df_arg[result_df_arg['cum_LOC'] <= cum_LOC_k_percent]
    buggy_commit = buggy_line_k_percent[buggy_line_k_percent['label']==1]
    recall_k_percent_effort = len(buggy_commit)/float(len(real_buggy_commits))

    return recall_k_percent_effort

def eval_metrics(result_df):

    pred = result_df['defective_commit_pred']
    y_test = result_df['label']

    prec, rec, f1, _ = precision_recall_fscore_support(y_test,pred,average='binary') # at threshold = 0.5
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
#     rec = tp/(tp+fn)

    FAR = fp/(fp+tn) # false alarm rate
    dist_heaven = math.sqrt((pow(1-rec,2)+pow(0-FAR,2))/2.0) # distance to heaven

    AUC = roc_auc_score(y_test, result_df['defective_commit_prob'])

    result_df['defect_density'] = result_df['defective_commit_prob']/result_df['LOC'] # predicted defect density
    result_df['actual_defect_density'] = result_df['label']/result_df['LOC'] #defect density

    result_df = result_df.sort_values(by='defect_density',ascending=False)
    actual_result_df = result_df.sort_values(by='actual_defect_density',ascending=False)
    actual_worst_result_df = result_df.sort_values(by='actual_defect_density',ascending=True)

    result_df['cum_LOC'] = result_df['LOC'].cumsum()
    actual_result_df['cum_LOC'] = actual_result_df['LOC'].cumsum()
    actual_worst_result_df['cum_LOC'] = actual_worst_result_df['LOC'].cumsum()

    real_buggy_commits = result_df[result_df['label'] == 1]

    label_list = list(result_df['label'])

    all_rows = len(label_list)

    # find Recall@20%Effort
    cum_LOC_20_percent = 0.2*result_df.iloc[-1]['cum_LOC']
    buggy_line_20_percent = result_df[result_df['cum_LOC'] <= cum_LOC_20_percent]
    buggy_commit = buggy_line_20_percent[buggy_line_20_percent['label']==1]
    recall_20_percent_effort = len(buggy_commit)/float(len(real_buggy_commits))

    # find Effort@20%Recall
    buggy_20_percent = real_buggy_commits.head(math.ceil(0.2 * len(real_buggy_commits)))
    buggy_20_percent_LOC = buggy_20_percent.iloc[-1]['cum_LOC']
    effort_at_20_percent_LOC_recall = int(buggy_20_percent_LOC) / float(result_df.iloc[-1]['cum_LOC'])

    # find P_opt
    percent_effort_list = []
    predicted_recall_at_percent_effort_list = []
    actual_recall_at_percent_effort_list = []
    actual_worst_recall_at_percent_effort_list = []

    for percent_effort in np.arange(10,101,10):
        predicted_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, result_df, real_buggy_commits)
        actual_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, actual_result_df, real_buggy_commits)
        actual_worst_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, actual_worst_result_df, real_buggy_commits)

        percent_effort_list.append(percent_effort/100)

        predicted_recall_at_percent_effort_list.append(predicted_recall_k_percent_effort)
        actual_recall_at_percent_effort_list.append(actual_recall_k_percent_effort)
        actual_worst_recall_at_percent_effort_list.append(actual_worst_recall_k_percent_effort)

    p_opt = 1 - ((auc(percent_effort_list, actual_recall_at_percent_effort_list) -
                 auc(percent_effort_list, predicted_recall_at_percent_effort_list)) /
                (auc(percent_effort_list, actual_recall_at_percent_effort_list) -
                auc(percent_effort_list, actual_worst_recall_at_percent_effort_list)))

    return prec, rec, f1, AUC, FAR, dist_heaven, recall_20_percent_effort, effort_at_20_percent_LOC_recall, p_opt
RF_data_dir = './data/'
def eval_result(proj_name,sampling_method = 'DE_SMOTE_min_df_3'):

    RF_result = pd.read_csv(RF_data_dir+proj_name+'_RF_'+sampling_method+'_prediction_result.csv')

    RF_result.columns = ['Unnamed', 'defective_commit_prob','defective_commit_pred','label','test_commit'] # for new result

    test_code, test_commit, test_label = prepare_data(proj_name, mode='test',
                                                              remove_python_common_tokens=remove_python_common_tokens)

    # get LOC of each commit
    RF_LOC = [len(code.splitlines()) for code in test_code]
    RF_df = pd.DataFrame()
    RF_df['commit_id'] = test_commit
    RF_df['LOC'] = RF_LOC

    RF_result = pd.merge(RF_df, RF_result,how='inner',left_on = 'commit_id', right_on='test_commit')
    prec, rec, f1, auc, FAR, dist_heaven, recall_20_percent_effort, effort_at_20_percent_LOC_recall,p_opt = eval_metrics(RF_result)


    # print('Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}, AUC: {:.2f}, FAR: {:.2f}, d2h: {:.2f}, PCI@20%LOC: {:.2f}, Effort@20%Recall: {:.2f}, POpt: {:.2f}'.format(prec, rec, f1, auc, FAR, dist_heaven, recall_20_percent_effort, effort_at_20_percent_LOC_recall,p_opt))

    print('Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}, AUC: {:.2f}'.format(prec, rec, f1, auc))

    return (prec, rec, f1, auc)


## start 
if __name__ == "main":



    create_path_if_not_exist('./data/')
    create_path_if_not_exist('./final_model/')

    # Run experiment 
    # qt_k_of_smote, qt_best_AUC_of_obj_func = run_experiment('qt')
    qt_k_of_smote, qt_best_AUC_of_obj_func = run_experiment('openstack')

    # print('The best k_neighbors of Qt:', qt_k_of_smote)



    # eval_result('qt')
    results = eval_result('openstack')

