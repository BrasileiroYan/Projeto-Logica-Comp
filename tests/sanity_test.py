from pysat.solvers import Minisat22

# ============================================================
# ENCODER: (¬r ⇒ (z ↔ AtMost1(ys)))
# ============================================================
def encode_atmost1_conditional(r, z, ys):
    clauses = []

    # (¬r ⇒ (z ⇒ AtMost1))  → (r ∨ ¬z ∨ ¬y_j ∨ ¬y_k)
    for j in range(len(ys)):
        for k in range(j + 1, len(ys)):
            clauses.append([ r, -z, -ys[j], -ys[k] ])

    # (¬r ⇒ (AtMost1 ⇒ z)) → (r ∨ z ∨ y_j ∨ y_k)
    clauses.append([r, z] + ys)

    # propagação unária: (¬r ⇒ (¬y_j ∨ z))
    for y in ys:
        clauses.append([ r, z, -y ])

    return clauses


# ============================================================
# FUNÇÃO PARA TESTAR UM CASO
# assumptions = [lit1, lit2, ...]
# (cada lit já é um literal: ex: -10, 1, 2, -3)
# ============================================================
def test_case(assumptions, clauses):
    solver = Minisat22()

    # adiciona CNF
    for c in clauses:
        solver.add_clause(c)

    # fixa literais
    for lit in assumptions:
        solver.add_clause([lit])

    sat = solver.solve()
    model = solver.get_model() if sat else None
    solver.delete()
    return sat, model


# ============================================================
# VARIÁVEIS
# ============================================================
r = 10
z = 11
ys = [1, 2, 3]

# gera cláusulas
clauses = encode_atmost1_conditional(r, z, ys)


# ============================================================
# TESTE A — AtMost1 verdadeiro (y1=1) → espera z=1
# r = 0, y1=1, y2=0, y3=0
# ============================================================
print("\n=== TESTE A: AtMost1 verdadeiro (y1=1) — espera z=1 ===")
assumA = [
    -r,   # r = 0
    1,    # y1 = 1
    -2,   # y2 = 0
    -3    # y3 = 0
]
sat, model = test_case(assumA, clauses)
print("SAT:", sat)
print("Modelo:", model)


# ============================================================
# TESTE B — AtMost1 falso (y1 = y2 = 1) → espera z=0 (ou UNSAT se propagação conflitar)
# r = 0, y1=1, y2=1, y3=0
# ============================================================
print("\n=== TESTE B: AtMost1 falso (y1=y2=1) — espera z=0 ou conflito ===")
assumB = [
    -r,   # r = 0
    1,    # y1 = 1
    2,    # y2 = 1
    -3    # y3 = 0
]
sat, model = test_case(assumB, clauses)
print("SAT:", sat)
print("Modelo:", model)


# ============================================================
# TESTE C — r = 1 (ignora tudo)
# ============================================================
print("\n=== TESTE C: r=1 (ignora tudo) ===")
assumC = [
    r,    # r = 1
    1,    # y1 = 1
    2,    # y2 = 1
    3,    # y3 = 1
    -z    # z = 0
]
sat, model = test_case(assumC, clauses)
print("SAT:", sat)
print("Modelo:", model)


# ============================================================
# TESTE D — Propagação: apenas y1=1 deve forçar z=1
# r = 0, y1=1
# ============================================================
print("\n=== TESTE D: propagação — y1=1 deve forçar z=1 ===")
assumD = [
    -r,   # r = 0
    1     # y1 = 1
]
sat, model = test_case(assumD, clauses)
print("SAT:", sat)
print("Modelo:", model)
# ============================================================
# TESTE E — AtMost1 verdadeiro (y1=0, y2=0, y3=0) → espera z=1
# r = 0, y1=0, y2=0, y3=0
# ============================================================
print("\n=== TESTE E: AtMost1 verdadeiro (todos 0) — espera z=1 ===")
assumE = [
    -r,  # r = 0
    -1,  # y1 = 0
    -2,  # y2 = 0
    -3   # y3 = 0
]
sat, model = test_case(assumE, clauses)
print("SAT:", sat)
print("Modelo:", model)
