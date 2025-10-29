import sys, os
if not sys.path[0] == os.path.abspath('.'):
    sys.path.insert(0, os.path.abspath('.'))

from models.imlib import IMLIB
from models.imlib_atmost1 import IMLIB_ATMOST1

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# training configurations
database_name = 'mushroom'  # alterar conforme o dataset
categorical_columns_index = list(range(22))  # exemplo
number_lines_per_partition = [8, 16]
max_rule_set_sizes = [1, 2, 3]
rules_accuracy_weights = [5, 10]
number_quantiles_ordinal_columns = 5
balance_instances = True
balance_instances_seed = 21
number_realizations = 1 # com number_realizations = 1 o tempo total estimado foi de 45 minutos

database_path = f'./databases/{database_name}.csv'

imlib_results_path = f'./drafts/tests/imlib_vs_imlib_atmost1/{database_name}_imlib.csv'
imlib_atmost1_results_path = f'./drafts/tests/imlib_vs_imlib_atmost1/{database_name}_imlib_atmost1.csv'

# cria a pasta para salvar os resultados, se não existir
os.makedirs('./drafts/tests/imlib_vs_imlib_atmost1', exist_ok=True)

# import dataset
Xy = pd.read_csv(database_path)
X = Xy.drop(['Class'], axis=1)
y = Xy['Class']

# dataframes que vão armazenar os resultados
columns = ['Configuration', 'Rules size', 'Rule set size', 'Sum rules size', 'Larger rule size', 'Accuracy', 'Training time']
imlib_results_df = pd.DataFrame([], columns=columns)
imlib_atmost1_results_df = pd.DataFrame([], columns=columns)

for lpp in tqdm(number_lines_per_partition, desc='lpp loop'):
    for mrss in tqdm(max_rule_set_sizes, desc='mrss loop'):
        for raw in tqdm(rules_accuracy_weights, desc='raw loop'):
            for r in tqdm(range(number_realizations), desc='realizations'):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                # inicializa os modelos
                imlib_model = IMLIB(
                    max_rule_set_size=mrss,
                    rules_accuracy_weight=raw,
                    categorical_columns_index=categorical_columns_index,
                    number_quantiles_ordinal_columns=number_quantiles_ordinal_columns,
                    number_lines_per_partition=lpp,
                    balance_instances=balance_instances,
                    balance_instances_seed=balance_instances_seed
                )

                imlib_atmost1_model = IMLIB_ATMOST1(
                    max_rule_set_size=mrss,
                    rules_accuracy_weight=raw,
                    categorical_columns_index=categorical_columns_index,
                    number_quantiles_ordinal_columns=number_quantiles_ordinal_columns,
                    number_lines_per_partition=lpp,
                    balance_instances=balance_instances,
                    balance_instances_seed=balance_instances_seed
                )

                # treina os modelos
                imlib_model.fit(X_train, y_train)
                imlib_atmost1_model.fit(X_train, y_train)

                # armazena resultados
                imlib_result = pd.DataFrame([[ 
                    f'lpp: {lpp} | mrss: {mrss} | raw: {raw}',
                    imlib_model.get_rules_size(),
                    imlib_model.get_rule_set_size(),
                    imlib_model.get_sum_rules_size(),
                    imlib_model.get_larger_rule_size(),
                    imlib_model.score(X_test, y_test),
                    imlib_model.get_total_time_solver_solutions()
                ]], columns=columns)

                imlib_atmost1_result = pd.DataFrame([[ 
                    f'lpp: {lpp} | mrss: {mrss} | raw: {raw}',
                    imlib_atmost1_model.get_rules_size(),
                    imlib_atmost1_model.get_rule_set_size(),
                    imlib_atmost1_model.get_sum_rules_size(),
                    imlib_atmost1_model.get_larger_rule_size(),
                    imlib_atmost1_model.score(X_test, y_test),
                    imlib_atmost1_model.get_total_time_solver_solutions()
                ]], columns=columns)

                imlib_results_df = pd.concat([imlib_results_df, imlib_result])
                imlib_atmost1_results_df = pd.concat([imlib_atmost1_results_df, imlib_atmost1_result])

# salva resultados
imlib_results_df.to_csv(imlib_results_path, index=False)
imlib_atmost1_results_df.to_csv(imlib_atmost1_results_path, index=False)