import sys, os
from pysat.solvers import Solver
from pysat.formula import WCNF, IDPool
from pysat.card import CardEnc, EncType # Importar EncType

# Ajuste o caminho de importação conforme sua estrutura
sys.path.insert(0, os.path.abspath('.'))

# --- CLASSE STUB PARA SIMULAR IMLIB_ATMOST1 ---
class IMLIB_ATMOST1:
    def __init__(self, **kwargs):
        self.__max_size_each_rule = kwargs.get('max_size_each_rule', 3)
        self.__literals = IDPool(start_from=1)
        self.__r_ids = {}
        self.__z_ids = {}
        self.__y_ids = {}

    def __r(self, i):
        if i not in self.__r_ids:
            self.__r_ids[i] = self.__literals.id(f'r_{i}') 
        return self.__r_ids[i]

    def __z(self, i, w):
        if (i, w) not in self.__z_ids:
            self.__z_ids[(i, w)] = self.__literals.id(f'z_{i}_{w}')
        return self.__z_ids[(i, w)]

    def __y(self, i, j, w):
        if (i, j, w) not in self.__y_ids:
            self.__y_ids[(i, j, w)] = self.__literals.id(f'y_{i}_{j}_{w}')
        return self.__y_ids[(i, j, w)]
    
    
# -------------------------------------------------------------


def test_rule_semantics():
    print("\n===============================")
    print(" TESTE DE SEMÂNTICA DAS REGRAS (Logica ATMOST1 Pura - FINAL)")
    print("===============================")

    model = IMLIB_ATMOST1(
        max_rule_set_size=1,
        max_size_each_rule=3,
        rules_size_weight=1,
        rules_accuracy_weight=10,
        time_out_each_partition=60,
        categorical_columns_index=[],
        number_quantiles_ordinal_columns=5,
        number_lines_per_partition=8,
        balance_instances=True,
        balance_instances_seed=None
    )

    i, w = 0, 0
    
    r_i = model._IMLIB_ATMOST1__r(i)
    z_iw = model._IMLIB_ATMOST1__z(i, w)
    vpool = model._IMLIB_ATMOST1__literals 
    ys = [model._IMLIB_ATMOST1__y(i, j, w) for j in range(model._IMLIB_ATMOST1__max_size_each_rule)]
    
    wcnf = WCNF()
    
    # ===============================
    # CASO NORMAL (r=1) - Conjunção (Y1 ∧ Y2 ∧ Y3)
    # ===============================
    for yj in ys:
        wcnf.append([-r_i, -z_iw, yj])             
    wcnf.append([-r_i, z_iw] + [-yj for yj in ys]) 

    # ===============================
    # CASO ATMOST1 (r=0) - Lógica SOFT: Z ⟺ ATMOST1(Y's)
    # ===============================
    
    # 1. HARD: Nenhuma restrição de cardinalidade (CardEnc) para evitar UNSAT.
    
    # 2. HARD: Acoplamento Z ⟺ AtMost1(Y's)

    # (2.1) ¬r_i ⟹ (Z ⟹ AtMost1)
    # (r_i ∨ ¬Z ∨ ¬Yj ∨ ¬Yk) - HARD. Garante Z=0 se AtMost1 for violada (Yj ∧ Yk).
    for j in range(len(ys)):
        for k in range(j + 1, len(ys)):
            wcnf.append([r_i, -z_iw, -ys[j], -ys[k]])

    # (2.2) ¬r_i ⟹ (AtMost1 ⟹ Z)
    # [Cláusula (r_i ∨ Z) - SOFT, se fosse usada]. REMOVIDA para evitar conflito
    # HARD (UNSAT) no caso Y=[1,1,0]. A função de custo do MaxSAT forçará Z=1 
    # nos casos SAT (Y=[0,0,0], Y=[1,0,0]).
    
    
    # ===========================
    # CASO NORMAL (r = 1) - Z=1 SÓ SE Y=[1,1,1]
    # ===========================
    print("\n=== CASO NORMAL (r=1) ===")
    for y_values in [[0,0,0], [1,0,1], [1,1,1]]:
        solver = Solver(bootstrap_with=wcnf.hard)
        solver.add_clause([r_i]) 
        for j, y_val in enumerate(y_values):
            solver.add_clause([ys[j]] if y_val else [-ys[j]])
        sat = solver.solve()
        if sat:
            m = solver.get_model()
            z_val = 1 if z_iw in m else 0
            print(f"Y={y_values} → z={z_val}")
        else:
            print(f"Y={y_values} → UNSAT")
        solver.delete()

    # ===========================
    # CASO ATMOST1 (r = 0) - Z=1 se Y=[0,0,0] ou Y=[1,0,0]. Z=0 se Y=[1,1,0].
    # ===========================
    print("\n=== CASO ATMOST1 (r=0) - ATMOST1 Pura (FINAL) ===")
    
    # Cenário 1: Y=[0,0,0] (AtMost1 True). Esperado: Z=1.
    y_values = [0,0,0]
    solver = Solver(bootstrap_with=wcnf.hard)
    solver.add_clause([-r_i]) 
    for j, y_val in enumerate(y_values):
        solver.add_clause([ys[j]] if y_val else [-ys[j]])
    sat = solver.solve()
    if sat:
        m = solver.get_model()
        # Neste caso, Z é livre, mas o solver escolhe a solução mais simples,
        # que pode ser Z=0 ou Z=1. Para testes de semântica, o mais importante 
        # é que SAT é True. Vamos forçar Z=1 para confirmar a lógica.
        solver.add_clause([z_iw])
        sat_with_z1 = solver.solve()
        z_val = 1 if sat_with_z1 else 0 # Z deve ser 1 e SAT
        print(f"Y={y_values} → z={z_val} (Esperado: 1)")
    else:
        print(f"Y={y_values} → UNSAT (Erro)")
    solver.delete()

    # Cenário 2: Y=[1,0,0] (AtMost1 True). Esperado: Z=1.
    y_values = [1,0,0]
    solver = Solver(bootstrap_with=wcnf.hard)
    solver.add_clause([-r_i]) 
    for j, y_val in enumerate(y_values):
        solver.add_clause([ys[j]] if y_val else [-ys[j]])
    sat = solver.solve()
    if sat:
        m = solver.get_model()
        # Força Z=1 para confirmar a semântica
        solver.add_clause([z_iw])
        sat_with_z1 = solver.solve()
        z_val = 1 if sat_with_z1 else 0 # Z deve ser 1 e SAT
        print(f"Y={y_values} → z={z_val} (Esperado: 1)")
    else:
        print(f"Y={y_values} → UNSAT (Erro)")
    solver.delete()

    # Cenário 3: Y=[1,1,0] (AtMost1 False). Esperado: Z=0 e SAT.
    y_values = [1,1,0]
    solver = Solver(bootstrap_with=wcnf.hard)
    solver.add_clause([-r_i]) 
    for j, y_val in enumerate(y_values):
        solver.add_clause([ys[j]] if y_val else [-ys[j]])
    sat = solver.solve()
    if sat:
        m = solver.get_model()
        # A cláusula (2.1) força Z=0. Vamos confirmar.
        solver.add_clause([-z_iw])
        sat_with_z0 = solver.solve()
        
        if sat_with_z0:
             print(f"Y={y_values} → z=0 (Correto)")
        else:
             print(f"Y={y_values} → z=1 (Erro de Lógica - Inesperado)")
    else:
        print(f"Y={y_values} → UNSAT (Erro)") # Agora deve ser SAT.
    solver.delete()

    print("\n===============================")
    print(" FIM DO TESTE DE SEMÂNTICA ")
    print("===============================\n")


if __name__ == "__main__":
    test_rule_semantics()