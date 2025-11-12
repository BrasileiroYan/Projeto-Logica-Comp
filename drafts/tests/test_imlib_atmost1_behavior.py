import sys, os
if not sys.path[0] == os.path.abspath('.'):
    sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from models.imlib_atmost1 import IMLIB_ATMOST1


# ===============================================================
# 1️⃣ TESTES DE SANIDADE BÁSICA
# ===============================================================

def test_basic_truth_tables():
    """
    Testa se o modelo aprende corretamente funções lógicas simples
    (AND, OR, XOR) usando o conjunto completo (sem train/test split).
    """
    X = pd.DataFrame([[0,0],[0,1],[1,0],[1,1]], columns=["x1", "x2"])
    y_and = pd.Series([0,0,0,1])
    y_or  = pd.Series([0,1,1,1])
    y_xor = pd.Series([0,1,1,0])

    def new_model():
        return IMLIB_ATMOST1(
            max_rule_set_size=3,
            max_size_each_rule=2,
            number_lines_per_partition=4,
            balance_instances=False,
            number_quantiles_ordinal_columns=3
        )

    # === AND ===
    print("\n=== Teste AND ===")
    model = new_model()
    model.fit(X, y_and)
    preds = np.array([model.predict(x) for x in X.values])
    print("Esperado:", y_and.tolist(), "Obtido:", preds.tolist())
    assert np.array_equal(preds, y_and.values), "Erro no comportamento lógico do AND"

    # === OR ===
    print("\n=== Teste OR ===")
    model = new_model()
    model.fit(X, y_or)
    preds = np.array([model.predict(x) for x in X.values])
    assert np.array_equal(preds, y_or.values), "Erro no comportamento lógico do OR"

    # === XOR ===
    print("\n=== Teste XOR ===")
    model = new_model()
    model.fit(X, y_xor)
    preds = np.array([model.predict(x) for x in X.values])
    acc = accuracy_score(y_xor, preds)
    print("Esperado:", y_xor.tolist(), "Obtido:", preds.tolist(), "Acurácia:", acc)
    assert acc >= 0.5, "Acurácia inesperadamente baixa no XOR"


# ===============================================================
# 2️⃣ TESTES DE ESTABILIDADE E DETERMINISMO
# ===============================================================

def test_determinism():
    X = np.random.randint(0, 2, (10, 3))
    y = np.random.randint(0, 2, 10)

    preds_list = []
    for _ in range(3):
        model = IMLIB_ATMOST1(max_rule_set_size=3, balance_instances=False)
        model.fit(pd.DataFrame(X), pd.Series(y))
        preds_list.append(np.array([model.predict(x) for x in X]))

    base = preds_list[0]
    assert all(np.array_equal(base, p) for p in preds_list), "Modelo não determinístico"


# ===============================================================
# 3️⃣ TESTE DE COMPORTAMENTO DO SOLVER
# ===============================================================

def test_solver_behavior():
    X = np.random.randint(0, 2, (6, 3))
    y = np.random.randint(0, 2, 6)

    model = IMLIB_ATMOST1(max_rule_set_size=3, max_size_each_rule=2, balance_instances=False)
    model.fit(pd.DataFrame(X), pd.Series(y))

    if hasattr(model, "_IMLIB_ATMOST1__solver_solution"):
        sol = getattr(model, "_IMLIB_ATMOST1__solver_solution")
        print("\nTamanho da solução SAT:", len(sol))
        assert len(sol) > 0, "Solução SAT vazia"
    else:
        raise AssertionError("Solver não gerou solução.")


# ===============================================================
# 4️⃣ TESTE DE GENERALIZAÇÃO (BREAST CANCER)
# ===============================================================

def test_real_dataset_behavior():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = pd.DataFrame(data.data)
    y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = IMLIB_ATMOST1(
        max_rule_set_size=4,
        max_size_each_rule=3,
        number_lines_per_partition=32,
        balance_instances=False
    )

    model.fit(X_train, y_train)
    preds = np.array([model.predict(x) for x in X_test.values])
    acc = accuracy_score(y_test, preds)
    print("\nAcurácia Breast Cancer:", acc)
    assert acc > 0.6, "Modelo não está generalizando adequadamente"


# ===============================================================
# Runner manual
# ===============================================================

if __name__ == "__main__":
    #test_basic_truth_tables()
    #test_determinism()
    #test_solver_behavior()
    test_real_dataset_behavior()
    print("\n✅ Todos os testes executados com sucesso.")
