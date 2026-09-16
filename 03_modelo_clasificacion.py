from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)


# ============================================================
# CARGA DE DATOS
# ============================================================

RUTA_DATOS = Path(
    "data/processed/dataset_modelo_fraude.csv"
)

CARPETA_OUTPUTS = Path("outputs")
CARPETA_GRAFICOS = Path("outputs/graficos")

CARPETA_OUTPUTS.mkdir(
    parents=True,
    exist_ok=True
)

CARPETA_GRAFICOS.mkdir(
    parents=True,
    exist_ok=True
)


df = pd.read_csv(
    RUTA_DATOS,
    parse_dates=["fecha_hora", "fecha"]
)

df = (
    df
    .sort_values("fecha_hora")
    .reset_index(drop=True)
)


# ============================================================
# VARIABLES DEL MODELO
# ============================================================

variable_objetivo = "fraude_confirmado"

variables_numericas = [
    "monto_soles",
    "dispositivo_nuevo",
    "operacion_internacional",
    "intentos_fallidos_24h",
    "transacciones_ultima_hora",
    "distancia_ubicacion_km",
    "hora",
    "horario_nocturno",
    "edad",
    "antiguedad_meses",
    "ingreso_mensual"
]

variables_categoricas = [
    "canal",
    "tipo_operacion",
    "categoria",
    "nivel_riesgo_cliente",
    "segmento",
    "departamento",
    "canal_preferido"
]

variables_modelo = (
    variables_numericas +
    variables_categoricas
)


# ============================================================
# DIVISIÓN TEMPORAL
# ============================================================

datos_entrenamiento = df[
    df["fecha_hora"] < "2025-09-01"
].copy()

datos_validacion = df[
    (df["fecha_hora"] >= "2025-09-01") &
    (df["fecha_hora"] < "2025-11-01")
].copy()

datos_prueba = df[
    df["fecha_hora"] >= "2025-11-01"
].copy()


print("Distribución temporal:")

print(
    "Entrenamiento:",
    datos_entrenamiento.shape
)

print(
    "Validación:",
    datos_validacion.shape
)

print(
    "Prueba:",
    datos_prueba.shape
)

print("\nTasa de fraude por conjunto:")

for nombre, datos in [
    ("Entrenamiento", datos_entrenamiento),
    ("Validación", datos_validacion),
    ("Prueba", datos_prueba)
]:
    print(
        nombre,
        round(
            100 * datos[variable_objetivo].mean(),
            2
        ),
        "%"
    )
    
    # ============================================================
# PREPARACIÓN DE LOS CONJUNTOS
# ============================================================

X_entrenamiento = datos_entrenamiento[variables_modelo]
y_entrenamiento = datos_entrenamiento[variable_objetivo]

X_validacion = datos_validacion[variables_modelo]
y_validacion = datos_validacion[variable_objetivo]

X_prueba = datos_prueba[variables_modelo]
y_prueba = datos_prueba[variable_objetivo]


# ============================================================
# PREPROCESAMIENTO
# ============================================================

procesamiento_numerico = Pipeline(
    steps=[
        (
            "imputacion",
            SimpleImputer(strategy="median")
        ),
        (
            "escalamiento",
            StandardScaler()
        )
    ]
)

procesamiento_categorico = Pipeline(
    steps=[
        (
            "imputacion",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "codificacion",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocesador = ColumnTransformer(
    transformers=[
        (
            "numericas",
            procesamiento_numerico,
            variables_numericas
        ),
        (
            "categoricas",
            procesamiento_categorico,
            variables_categoricas
        )
    ]
)


# ============================================================
# MODELO DE REGRESIÓN LOGÍSTICA
# ============================================================

modelo = Pipeline(
    steps=[
        (
            "preprocesamiento",
            preprocesador
        ),
        (
            "clasificador",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


print("\nEntrenando el modelo...")

modelo.fit(
    X_entrenamiento,
    y_entrenamiento
)

print("Modelo entrenado correctamente.")


# ============================================================
# PROBABILIDADES PREDICHAS
# ============================================================

probabilidad_validacion = modelo.predict_proba(
    X_validacion
)[:, 1]

probabilidad_prueba = modelo.predict_proba(
    X_prueba
)[:, 1]


# ============================================================
# MÉTRICAS GENERALES
# ============================================================

roc_auc_validacion = roc_auc_score(
    y_validacion,
    probabilidad_validacion
)

pr_auc_validacion = average_precision_score(
    y_validacion,
    probabilidad_validacion
)

roc_auc_prueba = roc_auc_score(
    y_prueba,
    probabilidad_prueba
)

pr_auc_prueba = average_precision_score(
    y_prueba,
    probabilidad_prueba
)

print("\nMétricas del modelo:")

print(
    "ROC AUC validación:",
    round(roc_auc_validacion, 4)
)

print(
    "PR AUC validación:",
    round(pr_auc_validacion, 4)
)

print(
    "ROC AUC prueba:",
    round(roc_auc_prueba, 4)
)

print(
    "PR AUC prueba:",
    round(pr_auc_prueba, 4)
)
# ============================================================
# UMBRAL CON LA MISMA CAPACIDAD DE ALERTAS QUE LAS REGLAS
# ============================================================

tasa_alertas_reglas = datos_validacion[
    "alerta_generada"
].mean()

umbral_modelo = np.quantile(
    probabilidad_validacion,
    1 - tasa_alertas_reglas
)

print("\nSelección del umbral:")

print(
    "Tasa de alertas de las reglas en validación:",
    round(100 * tasa_alertas_reglas, 2),
    "%"
)

print(
    "Umbral seleccionado para el modelo:",
    round(umbral_modelo, 4)
)


# ============================================================
# FUNCIÓN DE EVALUACIÓN OPERATIVA
# ============================================================

def calcular_metricas(
    datos,
    prediccion,
    nombre_metodo
):
    real = datos[
        variable_objetivo
    ].to_numpy()

    prediccion = np.asarray(
        prediccion
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        real,
        prediccion,
        labels=[0, 1]
    ).ravel()

    total_alertas = tp + fp
    total_fraudes = tp + fn

    precision = (
        tp / total_alertas
        if total_alertas > 0
        else 0
    )

    recall = (
        tp / total_fraudes
        if total_fraudes > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0
    )

    montos = datos[
        "monto_soles"
    ].to_numpy()

    monto_prevenido = montos[
        (real == 1) &
        (prediccion == 1)
    ].sum()

    perdida_fraude = montos[
        (real == 1) &
        (prediccion == 0)
    ].sum()

    return {
        "metodo": nombre_metodo,
        "transacciones": len(datos),
        "alertas": int(total_alertas),
        "fraudes_confirmados": int(total_fraudes),
        "fraudes_detectados": int(tp),
        "falsos_positivos": int(fp),
        "fraudes_no_detectados": int(fn),
        "tasa_alertas_pct": round(
            100 * total_alertas / len(datos),
            2
        ),
        "precision_pct": round(
            100 * precision,
            2
        ),
        "recall_pct": round(
            100 * recall,
            2
        ),
        "f1_pct": round(
            100 * f1,
            2
        ),
        "monto_prevenido": round(
            monto_prevenido,
            2
        ),
        "perdida_por_fraude": round(
            perdida_fraude,
            2
        )
    }


# ============================================================
# COMPARACIÓN FINAL EN EL CONJUNTO DE PRUEBA
# ============================================================

prediccion_reglas = datos_prueba[
    "alerta_generada"
].astype(int)

prediccion_modelo = (
    probabilidad_prueba >= umbral_modelo
).astype(int)

metricas_reglas = calcular_metricas(
    datos_prueba,
    prediccion_reglas,
    "Reglas actuales"
)

metricas_modelo = calcular_metricas(
    datos_prueba,
    prediccion_modelo,
    "Modelo logístico"
)

comparacion = pd.DataFrame(
    [
        metricas_reglas,
        metricas_modelo
    ]
)

print(
    "\nComparación final en prueba:"
)

print(
    comparacion.to_string(
        index=False
    )
)

ruta_comparacion = (
    CARPETA_OUTPUTS /
    "comparacion_reglas_vs_modelo.csv"
)

comparacion.to_csv(
    ruta_comparacion,
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nComparación guardada en:"
)

print(ruta_comparacion)
# ============================================================
# BASE DE RESULTADOS PARA POSTGRESQL Y POWER BI
# ============================================================

resultados_prueba = datos_prueba.copy()

resultados_prueba[
    "probabilidad_modelo"
] = probabilidad_prueba

resultados_prueba[
    "prediccion_modelo"
] = prediccion_modelo

resultados_prueba[
    "prediccion_reglas"
] = prediccion_reglas.to_numpy()

resultados_prueba[
    "prioridad_modelo"
] = resultados_prueba[
    "probabilidad_modelo"
].rank(
    method="first",
    ascending=False
).astype(int)

resultados_prueba[
    "resultado_modelo"
] = np.select(
    [
        (
            resultados_prueba["prediccion_modelo"] == 1
        ) &
        (
            resultados_prueba["fraude_confirmado"] == 1
        ),
        (
            resultados_prueba["prediccion_modelo"] == 1
        ) &
        (
            resultados_prueba["fraude_confirmado"] == 0
        ),
        (
            resultados_prueba["prediccion_modelo"] == 0
        ) &
        (
            resultados_prueba["fraude_confirmado"] == 1
        )
    ],
    [
        "Verdadero positivo",
        "Falso positivo",
        "Falso negativo"
    ],
    default="Verdadero negativo"
)

ruta_predicciones = (
    CARPETA_OUTPUTS /
    "predicciones_modelo_prueba.csv"
)

resultados_prueba.to_csv(
    ruta_predicciones,
    index=False,
    encoding="utf-8-sig"
)

diferencia_prevenida = (
    metricas_modelo["monto_prevenido"] -
    metricas_reglas["monto_prevenido"]
)

reduccion_perdida_pct = (
    100 * diferencia_prevenida /
    metricas_reglas["perdida_por_fraude"]
)

print("\nImpacto económico del modelo:")

print(
    "Monto adicional prevenido: S/",
    round(diferencia_prevenida, 2)
)

print(
    "Reducción porcentual de pérdida:",
    round(reduccion_perdida_pct, 2),
    "%"
)

print(
    "\nPredicciones guardadas en:"
)

print(ruta_predicciones)
# ============================================================
# ESTRATEGIA HÍBRIDA: REGLAS + MODELO
# ============================================================

def seleccionar_alertas_por_capacidad(
    puntajes,
    numero_alertas
):
    puntajes = np.asarray(puntajes)

    posiciones_ordenadas = np.argsort(
        -puntajes
    )

    alertas = np.zeros(
        len(puntajes),
        dtype=int
    )

    alertas[
        posiciones_ordenadas[:numero_alertas]
    ] = 1

    return alertas


numero_alertas_validacion = int(
    datos_validacion[
        "alerta_generada"
    ].sum()
)

resultados_pesos = []

for peso_modelo in np.arange(
    0,
    1.01,
    0.05
):
    peso_reglas = 1 - peso_modelo

    puntaje_hibrido_validacion = (
        peso_modelo *
        probabilidad_validacion
        +
        peso_reglas *
        (
            datos_validacion[
                "puntaje_riesgo"
            ].to_numpy() / 100
        )
    )

    alerta_hibrida_validacion = (
        seleccionar_alertas_por_capacidad(
            puntaje_hibrido_validacion,
            numero_alertas_validacion
        )
    )

    metricas = calcular_metricas(
        datos_validacion,
        alerta_hibrida_validacion,
        "Estrategia híbrida"
    )

    resultados_pesos.append(
        {
            "peso_modelo": round(
                peso_modelo,
                2
            ),
            "peso_reglas": round(
                peso_reglas,
                2
            ),
            **metricas
        }
    )


evaluacion_pesos = pd.DataFrame(
    resultados_pesos
)

mejor_configuracion = (
    evaluacion_pesos
    .sort_values(
        [
            "monto_prevenido",
            "fraudes_detectados"
        ],
        ascending=False
    )
    .iloc[0]
)

mejor_peso_modelo = float(
    mejor_configuracion[
        "peso_modelo"
    ]
)

mejor_peso_reglas = float(
    mejor_configuracion[
        "peso_reglas"
    ]
)

print(
    "\nMejor combinación encontrada en validación:"
)

print(
    "Peso del modelo:",
    mejor_peso_modelo
)

print(
    "Peso de las reglas:",
    mejor_peso_reglas
)

print(
    "Monto prevenido en validación: S/",
    mejor_configuracion[
        "monto_prevenido"
    ]
)


# ============================================================
# EVALUACIÓN HÍBRIDA EN PRUEBA
# ============================================================

numero_alertas_prueba = round(
    tasa_alertas_reglas *
    len(datos_prueba)
)

puntaje_hibrido_prueba = (
    mejor_peso_modelo *
    probabilidad_prueba
    +
    mejor_peso_reglas *
    (
        datos_prueba[
            "puntaje_riesgo"
        ].to_numpy() / 100
    )
)

prediccion_hibrida = (
    seleccionar_alertas_por_capacidad(
        puntaje_hibrido_prueba,
        numero_alertas_prueba
    )
)

metricas_hibridas = calcular_metricas(
    datos_prueba,
    prediccion_hibrida,
    "Estrategia híbrida"
)

comparacion_final = pd.DataFrame(
    [
        metricas_reglas,
        metricas_modelo,
        metricas_hibridas
    ]
)

print(
    "\nComparación definitiva en prueba:"
)

print(
    comparacion_final.to_string(
        index=False
    )
)

comparacion_final.to_csv(
    CARPETA_OUTPUTS /
    "comparacion_final_estrategias.csv",
    index=False,
    encoding="utf-8-sig"
)

evaluacion_pesos.to_csv(
    CARPETA_OUTPUTS /
    "evaluacion_pesos_hibridos.csv",
    index=False,
    encoding="utf-8-sig"
)


# Agregar resultado híbrido al archivo de predicciones

resultados_prueba[
    "puntaje_hibrido"
] = puntaje_hibrido_prueba

resultados_prueba[
    "prediccion_hibrida"
] = prediccion_hibrida

resultados_prueba.to_csv(
    ruta_predicciones,
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nResultados híbridos guardados en:"
)

print(
    CARPETA_OUTPUTS /
    "comparacion_final_estrategias.csv"
)

print(
    CARPETA_OUTPUTS /
    "evaluacion_pesos_hibridos.csv"
)

print(ruta_predicciones)

# ============================================================
# ARCHIVO REDUCIDO PARA POSTGRESQL
# ============================================================

columnas_postgresql = [
    "transaccion_id",
    "probabilidad_modelo",
    "prediccion_reglas",
    "prediccion_modelo",
    "resultado_modelo",
    "prioridad_modelo",
    "puntaje_hibrido",
    "prediccion_hibrida"
]

predicciones_postgresql = resultados_prueba[
    columnas_postgresql
].copy()

ruta_postgresql = (
    CARPETA_OUTPUTS /
    "predicciones_modelo_postgresql.csv"
)

predicciones_postgresql.to_csv(
    ruta_postgresql,
    index=False,
    encoding="utf-8"
)

print(
    "\nArchivo preparado para PostgreSQL:"
)

print(ruta_postgresql)

print(
    "Registros:",
    len(predicciones_postgresql)
)