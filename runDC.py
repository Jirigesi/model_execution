# move and rename splitted files into specific folder 

# train new model multiple times and log results especially precision, recall, F1, AUC 

# run each model 10 times and save results into an csv file 
from JITLine_replication_package.JITLine.JITLine_run import create_path_if_not_exist, run_experiment, eval_result
import os 
import subprocess
import csv



# os.chdir("/home/fjiriges/model_execution")
# characters= ["Filecount", "Editcount", "MultilineCommentscount"]
characters= ["MultilineCommentscount"]
levels = [ "hard", "easy"]

for character in characters:
    for level in levels:
        os.chdir("/home/fjiriges/model_execution")
        # Each time must copy correspoding training and testing data
        test_file_to_move = "splittedData/" + level + "_" + character + "_openstack_test.pkl"
        train_file_to_move = "splittedData/" + level + "_" + character + "_openstack_train.pkl"
        # splittedData / easy_Filecount_openstack_train.pkl

        subprocess.call(["mv", test_file_to_move, "JITLine_replication_package/JITLine/data/openstack_test.pkl"])
        print("moved testing files")

        os.chdir("/home/fjiriges/model_execution")
        subprocess.call(["mv", train_file_to_move, "JITLine_replication_package/JITLine/data/openstack_train.pkl"])
        print("moved training files")

        os.chdir("/home/fjiriges/model_execution/JITLine_replication_package/JITLine")
        for i in range(3):
            ######
            print("################")
            print("-----Runing ",i, character, level, " model-------")
            projects = ['openstack', 'qt']
            sampling_methods = 'DE_SMOTE_min_df_3'
            remove_python_common_tokens = True
            # os.chdir("JITLine_replication_package/JITLine")
            #
            # create_path_if_not_exist('./data/')
            # create_path_if_not_exist('./final_model/')

            # Run experiment
            # qt_k_of_smote, qt_best_AUC_of_obj_func = run_experiment('qt')
            qt_k_of_smote, qt_best_AUC_of_obj_func = run_experiment('openstack')

            # print('The best k_neighbors of Qt:', qt_k_of_smote)

            RF_data_dir = './data/'

            # eval_result('qt')
            results = eval_result('openstack')
            # store results

            with open('DCresults.csv', mode='a+') as csv_file:
                # fieldnames = ["prec", "rec", "f1", "auc"]
                csvwriter = csv.writer(csv_file)
                # writer.writeheader()
                csvwriter.writerow([level, character, i, results[0], results[1], results[2], results[3]])

            # move result to a collection folder 
            result_new_name = str(i) + "_" + level + "_" + character

            subprocess.call(["mv", "/home/fjiriges/model_execution/JITLine_replication_package/JITLine/data/openstack_RF_DE_SMOTE_min_df_3_prediction_result.csv",
                             "/home/fjiriges/model_execution/JITLine_replication_package/result_collctor/" + result_new_name])