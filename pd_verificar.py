"""Comprueba la parte de la PD contra resultados que se saben de antemano.

No entrena nada: usa los modelos de modelos/ y tarda menos de un minuto. Hay que correrlo en un
clon limpio antes de la sustentacion. Termina con error si algo falla.

    python pd_verificar.py
"""

import subprocess
import sys

import numpy as np
from scipy import stats

from pd_figuras import partes_evaluacion
from riesgo import configuracion, metricas, predecir_pd, scorecard

# Se corre en un proceso aparte porque este script si carga PyTorch. El buscador hace fallar
# cualquier "import torch". No sirve poner sys.modules["torch"] = None: scipy revisa esa entrada
# aunque nunca use torch, y se cae por la prueba y no por nuestro codigo.
SIN_TORCH = """
import sys
class SinTorch:
    def find_spec(self, nombre, ruta=None, objetivo=None):
        if nombre.split(".")[0] == "torch":
            raise ImportError("torch esta bloqueado en esta prueba")
sys.meta_path.insert(0, SinTorch())
import warnings
warnings.filterwarnings("error", message="evaluar_cliente")
from riesgo import predecir_pd
modelos = predecir_pd.cargar("modelos")
r = predecir_pd.evaluar_cliente(modelos, {
    "loan_amnt": 10000, "plazo_meses": 36, "antiguedad_empleo": 5, "annual_inc": 60000, "dti": 15,
    "delinq_2yrs": 0, "antiguedad_credito_meses": 180, "inq_last_6mths": 1, "meses_desde_mora": None,
    "meses_desde_registro": None, "open_acc": 10, "pub_rec": 0, "revol_bal": 10000, "revol_util": 50,
    "total_acc": 20, "collections_12_mths_ex_med": 0, "acc_now_delinq": 0, "home_ownership": "RENT",
    "verification_status": "Not Verified", "purpose": "debt_consolidation", "addr_state": "CA",
    "initial_list_status": "f"})
assert "torch" not in sys.modules
print(round(r["pd"], 4), round(r["puntaje"], 1), round(r["percentil"], 1))
"""

fallas = []


def revisar(descripcion, condicion, detalle=""):
    print(f"[{'OK' if condicion else 'FALLA'}] {descripcion} {detalle}")
    if not condicion:
        fallas.append(descripcion)


def metricas_a_mano():
    """Caso de 4 creditos, calculado en papel."""
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    # 3 de los 4 pares (malo, bueno) estan bien ordenados: AUC = 3/4
    revisar("AUC a mano = 0.75", np.isclose(metricas.auc(y, p), 0.75))
    revisar("Gini a mano = 0.5", np.isclose(metricas.gini(y, p), 0.5))
    # con umbral 0.35: TPR = 1 y FPR = 0.5
    revisar("KS a mano = 0.5", np.isclose(metricas.ks(y, p), 0.5))
    revisar("Brier a mano", np.isclose(metricas.brier(y, p), (0.01 + 0.16 + 0.4225 + 0.04) / 4))


def metricas_contra_otra_formula(y, p):
    """AUC contra el conteo de Mann-Whitney y KS contra scipy, sobre datos reales."""
    malos = y == 1
    rangos = stats.rankdata(p)
    auc_mw = (rangos[malos].sum() - malos.sum() * (malos.sum() + 1) / 2) / (malos.sum() * (~malos).sum())
    revisar("AUC igual a Mann-Whitney", np.isclose(metricas.auc(y, p), auc_mw))
    ks_scipy = stats.ks_2samp(p[malos], p[~malos]).statistic
    revisar("KS igual a scipy", np.isclose(metricas.ks(y, p), ks_scipy))


def formula_scorecard(par):
    base, odds, pdo = par["puntaje_base"], par["odds_referencia"], par["pdo"]
    revisar("odds de referencia dan el puntaje base", np.isclose(scorecard.puntaje(1 / (1 + odds), par), base))
    revisar("duplicar las odds suma PDO", np.isclose(scorecard.puntaje(1 / (1 + 2 * odds), par), base + pdo))
    revisar("reducir las odds a la mitad resta PDO", np.isclose(scorecard.puntaje(1 / (1 + odds / 2), par), base - pdo))
    ps = np.array([0.01, 0.1, 0.3, 0.7])
    revisar("probabilidad(puntaje(p)) = p", np.allclose(scorecard.probabilidad(scorecard.puntaje(ps, par), par), ps))
    revisar("mas PD da menos puntaje", np.all(np.diff(scorecard.puntaje(ps, par)) < 0))


def red_numpy_igual_a_torch(modelos, X):
    from riesgo import red as redes
    con_torch = redes.predecir(redes.cargar("modelos/pd_red.pt"), X)
    diferencia = np.abs(con_torch - predecir_pd.predecir_matriz(modelos, X, "red")).max()
    revisar("predecir_numpy igual a PyTorch", diferencia < 1e-5, f"(diferencia maxima {diferencia:.1e})")


def puntos(modelos, prueba):
    clientes = prueba.sample(50, random_state=1).to_dict("records")
    par = modelos["scorecard"]
    errores = []
    for c in clientes:
        r = predecir_pd.evaluar_cliente(modelos, c)
        errores.append(abs(r["puntos_base"] + sum(r["puntos"].values()) - r["puntaje"]))
    revisar("puntos_base + suma(puntos) = puntaje", max(errores) < 0.01, f"(error maximo {max(errores):.1e})")

    primero = predecir_pd.evaluar_cliente(modelos, clientes[0])
    revisar("mismo cliente, mismos puntos", primero == predecir_pd.evaluar_cliente(modelos, clientes[0]))

    # Con un modelo lineal, los valores de Shapley son exactamente coeficiente * (x - referencia).
    # Si el algoritmo de permutaciones estuviera mal, aqui no coincidirian.
    x = predecir_pd.transformar(modelos, prueba.head(1))[0]
    directos = scorecard.puntos_logistica(modelos["logistica"], x, modelos["referencia"], modelos["grupos"], par)
    por_orden = scorecard.puntos_permutacion(
        lambda X: predecir_pd.predecir_matriz(modelos, X, "logistica"), x, modelos["referencia"],
        modelos["grupos"], par, 30, modelos["semilla"])
    error = max(abs(directos[v] - por_orden[v]) for v in directos)
    revisar("permutaciones = coeficientes en la logistica", error < 1e-3, f"(error maximo {error:.1e})")

    # Con permutaciones aleatorias hay ruido de muestreo: se mide contra una corrida mucho mas larga.
    def con_red(m):
        return scorecard.puntos_permutacion(lambda X: predecir_pd.predecir_matriz(modelos, X, "red"), x,
                                            modelos["referencia"], modelos["grupos"], par, m, modelos["semilla"])
    ruido = max(abs(con_red(par["permutaciones"])[v] - con_red(4000)[v]) for v in directos)
    revisar(f"{par['permutaciones']} permutaciones bastan (menos de 1 punto)", ruido < 1, f"(diferencia maxima {ruido:.2f})")


def consistencia(modelos, entrenamiento, prueba):
    fila = prueba.iloc[[0]]
    directo = float(predecir_pd.probabilidad(modelos, fila)[0])
    por_dic = predecir_pd.evaluar_cliente(modelos, fila.iloc[0].to_dict())["pd"]
    revisar("evaluar_cliente coincide con probabilidad", np.isclose(directo, por_dic, atol=1e-6))

    calculados = scorecard.puntaje(predecir_pd.probabilidad(modelos, entrenamiento), modelos["scorecard"])
    revisar("pd_puntajes.npy es de este modelo y estos parametros",
            np.allclose(np.sort(calculados), modelos["puntajes"], atol=1e-2))
    mediana = np.median(modelos["puntajes"])
    por_percentil = 100 * np.searchsorted(modelos["puntajes"], mediana, side="right") / len(modelos["puntajes"])
    revisar("la mediana queda cerca del percentil 50", abs(por_percentil - 50) < 1, f"({por_percentil:.1f})")


def main():
    a = configuracion.argumentos(__doc__).parse_args()
    cfg = configuracion.cargar(a.config, a.cambios)
    entrenamiento, validacion, prueba = partes_evaluacion(cfg, False)
    modelos = predecir_pd.cargar("modelos")

    revisar("particiones del HANDOFF", [len(entrenamiento), len(validacion), len(prueba)] == [130832, 32709, 104989])
    revisar("todos los creditos de prueba son de 2014-2015", set(prueba["anio"]) == {2014, 2015})
    revisar("validacion estratificada", abs(validacion["malo"].mean() - entrenamiento["malo"].mean()) < 1e-3)

    metricas_a_mano()
    metricas_contra_otra_formula(prueba["malo"].to_numpy(), predecir_pd.probabilidad(modelos, prueba))
    formula_scorecard(modelos["scorecard"])
    red_numpy_igual_a_torch(modelos, predecir_pd.transformar(modelos, prueba))
    puntos(modelos, prueba)
    consistencia(modelos, entrenamiento, prueba)

    resultado = subprocess.run([sys.executable, "-c", SIN_TORCH], capture_output=True, text=True)
    revisar("evaluar_cliente funciona sin PyTorch", resultado.returncode == 0,
            f"(pd, puntaje, percentil: {resultado.stdout.strip()}) {resultado.stderr[-300:] if resultado.returncode else ''}")

    if fallas:
        sys.exit(f"{len(fallas)} comprobaciones fallaron.")
    print("Todas las comprobaciones pasaron.")


if __name__ == "__main__":
    main()
