import sys, os
if not sys.path[0] == os.path.abspath('.'):
    sys.path.insert(0, os.path.abspath('.'))

from models.imlib import IMLIB
from models.imlib_atmost1 import IMLIB_ATMOST1

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.stats import entropy
from tqdm import tqdm


# ===============================================================
# DATASET CONFIG
# ===============================================================

database_names = [
    'lung_cancer', 'iris', 'parkinsons', 'ionosphere', 'wdbc',
    'transfusion', 'pima', 'titanic', 'depressed', 'mushroom', 'twitter'
]

categorical_columns_indexes = [
    [0, 1], [], [], [0], [],
    [], [0], [0, 2, 3, 5], [6],
    list(range(22)), [42, 43, 44, 45, 46, 47, 48]
]


# ===============================================================
# MODEL CONFIG
# ===============================================================

number_lines_per_partition = [8, 16]
max_rule_set_sizes = [1, 2, 3]
max_sizes_each_rule = [1, 2, 3]
rules_size_weight = 1
rules_accuracy_weights = [5, 10]
number_quantiles_ordinal_columns = 5
balance_instances = True
balance_instances_seed = 21

# number of runs
number_realizations = 10


# ===============================================================
# MAIN LOOP PER DATASET
# ===============================================================

for database_name, categorical_columns_index in zip(database_names, categorical_columns_indexes):

    print(f"\n\n============================")
    print(f" PROCESSING DATASET: {database_name}")
    print(f"============================\n")

    # result tables
    columns = [
        'Configuration', 'Rule sizes', 'Average deviation of rule sizes',
        'Standard deviation of rule sizes', 'Entropy of rule sizes',
        'Number of rules', '|R|', 'Largest rule size',
        'Accuracy', 'Training time', 'Confusion matrix'
    ]

    imlib_results_df = pd.DataFrame([], columns=columns)
    imlib_atmost1_results_df = pd.DataFrame([], columns=columns)

    # save paths
    base_path = './drafts/tests_2/imlib_vs_imlib_atmost1_results'
    imlib_results_path = f'{base_path}/{database_name}_imlib.csv'
    imlib_atmost1_results_path = f'{base_path}/{database_name}_imlib_atmost1.csv'

    # load dataset
    database_path = f'./databases/{database_name}.csv'
    Xy = pd.read_csv(database_path)
    X = Xy.drop(['Class'], axis=1)
    y = Xy['Class']


    # ===============================================================
    # REALIZATIONS LOOP
    # ===============================================================

    for r in tqdm(range(number_realizations), desc=f"{database_name} realizations"):

        # train & test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        # validation split
        X_train_train, X_val, y_train_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1
        )

        # best configs: [lpp, mrss, raw, mser, accuracy]
        imlib_best_config = [0, 0, 0, 0, 0]
        imlib_atmost1_best_config = [0, 0, 0, 0, 0]


        # ===============================================================
        # GRID SEARCH
        # ===============================================================

        for lpp in tqdm(number_lines_per_partition, desc="lpp loop"):

            # avoid division by zero
            if lpp == 0:
                continue

            for mrss in max_rule_set_sizes:
                for raw in rules_accuracy_weights:
                    for mser in range(1, mrss + 1):

                        # =======================
                        # IMLIB model
                        # =======================
                        imlib_model = IMLIB(
                            max_rule_set_size=mrss,
                            max_size_each_rule=mser,
                            rules_accuracy_weight=raw,
                            categorical_columns_index=categorical_columns_index,
                            number_quantiles_ordinal_columns=number_quantiles_ordinal_columns,
                            number_lines_per_partition=lpp,
                            balance_instances=balance_instances,
                            balance_instances_seed=balance_instances_seed
                        )

                        # =======================
                        # IMLIB ATMOST1 model
                        # =======================
                        imlib_atmost1_model = IMLIB_ATMOST1(
                            max_rule_set_size=mrss,
                            max_size_each_rule=mser,
                            rules_accuracy_weight=raw,
                            categorical_columns_index=categorical_columns_index,
                            number_quantiles_ordinal_columns=number_quantiles_ordinal_columns,
                            number_lines_per_partition=lpp,
                            balance_instances=balance_instances,
                            balance_instances_seed=balance_instances_seed
                        )

                        # fit
                        imlib_model.fit(X_train_train, y_train_train)
                        imlib_atmost1_model.fit(X_train_train, y_train_train)

                        # validation accuracy
                        acc_imlib = imlib_model.score(X_val, y_val)
                        acc_at1 = imlib_atmost1_model.score(X_val, y_val)

                        # update best configs
                        if acc_imlib > imlib_best_config[-1]:
                            imlib_best_config = [lpp, mrss, raw, mser, acc_imlib]

                        if acc_at1 > imlib_atmost1_best_config[-1]:
                            imlib_atmost1_best_config = [lpp, mrss, raw, mser, acc_at1]


        # ===============================================================
        # RETRAIN BEST CONFIG
        # ===============================================================

        imlib_model = IMLIB(
            max_rule_set_size=imlib_best_config[1],
            max_size_each_rule=imlib_best_config[3],
            rules_accuracy_weight=imlib_best_config[2],
            categorical_columns_index=categorical_columns_index,
            number_quantiles_ordinal_columns=number_quantiles_ordinal_columns,
            number_lines_per_partition=imlib_best_config[0],
            balance_instances=balance_instances,
            balance_instances_seed=balance_instances_seed
        )

        imlib_atmost1_model = IMLIB_ATMOST1(
            max_rule_set_size=imlib_atmost1_best_config[1],
            max_size_each_rule=imlib_atmost1_best_config[3],
            rules_accuracy_weight=imlib_atmost1_best_config[2],
            categorical_columns_index=categorical_columns_index,
            number_quantiles_ordinal_columns=number_quantiles_ordinal_columns,
            number_lines_per_partition=imlib_atmost1_best_config[0],
            balance_instances=balance_instances,
            balance_instances_seed=balance_instances_seed
        )

        # final training
        imlib_model.fit(X_train, y_train)
        imlib_atmost1_model.fit(X_train, y_train)


        # ===============================================================
        # RESULTS — IMLIB
        # ===============================================================

        imlib_rules_size = imlib_model.get_rules_size()
        total_size = sum(imlib_rules_size)
        imlib_result = pd.DataFrame([[
            f'lpp: {imlib_best_config[0]} | mrss: {imlib_best_config[1]} | raw: {imlib_best_config[2]} | mser: {imlib_best_config[3]}',
            imlib_rules_size,
            np.mean(imlib_rules_size),
            np.std(imlib_rules_size),
            entropy([s / total_size for s in imlib_rules_size], base=2),
            imlib_model.get_rule_set_size(),
            imlib_model.get_sum_rules_size(),
            imlib_model.get_larger_rule_size(),
            imlib_model.score(X_test, y_test),
            imlib_model.get_total_time_solver_solutions(),
            confusion_matrix(y_test, [imlib_model.predict(x) for x in X_test.values])
        ]], columns=columns)

        imlib_results_df = pd.concat([imlib_results_df, imlib_result])


        # ===============================================================
        # RESULTS — IMLIB ATMOST1
        # ===============================================================

        at1_rules_size = imlib_atmost1_model.get_rules_size()
        total_size = sum(at1_rules_size)

        imlib_at1_result = pd.DataFrame([[
            f'lpp: {imlib_atmost1_best_config[0]} | mrss: {imlib_atmost1_best_config[1]} | raw: {imlib_atmost1_best_config[2]} | mser: {imlib_atmost1_best_config[3]}',
            at1_rules_size,
            np.mean(at1_rules_size),
            np.std(at1_rules_size),
            entropy([s / total_size for s in at1_rules_size], base=2),
            imlib_atmost1_model.get_rule_set_size(),
            imlib_atmost1_model.get_sum_rules_size(),
            imlib_atmost1_model.get_larger_rule_size(),
            imlib_atmost1_model.score(X_test, y_test),
            imlib_atmost1_model.get_total_time_solver_solutions(),
            confusion_matrix(y_test, [imlib_atmost1_model.predict(x) for x in X_test.values])
        ]], columns=columns)

        imlib_atmost1_results_df = pd.concat([imlib_atmost1_results_df, imlib_at1_result])


    # ===============================================================
    # SUMMARY ROWS (AVERAGES)
    # ===============================================================

    def summarize(df):
        return pd.DataFrame([[
            'Averages', '',
            f"{df['Average deviation of rule sizes'].mean():.4f} ± {df['Average deviation of rule sizes'].std():.4f}",
            f"{df['Standard deviation of rule sizes'].mean():.4f} ± {df['Standard deviation of rule sizes'].std():.4f}",
            f"{df['Entropy of rule sizes'].mean():.4f} ± {df['Entropy of rule sizes'].std():.4f}",
            f"{df['Number of rules'].mean():.4f} ± {df['Number of rules'].std():.4f}",
            f"{df['|R|'].mean():.4f} ± {df['|R|'].std():.4f}",
            f"{df['Largest rule size'].mean():.4f} ± {df['Largest rule size'].std():.4f}",
            f"{df['Accuracy'].mean():.4f} ± {df['Accuracy'].std():.4f}",
            f"{df['Training time'].mean():.4f} ± {df['Training time'].std():.4f}",
            ''
        ]], columns=columns)

    imlib_results_df = pd.concat([imlib_results_df, summarize(imlib_results_df)])
    imlib_atmost1_results_df = pd.concat([imlib_atmost1_results_df, summarize(imlib_atmost1_results_df)])

    imlib_results_df.to_csv(imlib_results_path, index=False)
    imlib_atmost1_results_df.to_csv(imlib_atmost1_results_path, index=False)
