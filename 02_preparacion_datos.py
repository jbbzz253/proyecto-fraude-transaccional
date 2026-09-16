from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# RUTAS DE ENTRADA Y SALIDA
# ============================================================

# Periodo de la base ficticia
FECHA_INICIAL = "2025-01-01"
FECHA_FINAL = "2025-12-31"

RUTA_CLIENTES = Path(
    "data/raw/clientes_ficticios.csv"
)

RUTA_TRANSACCIONES = Path(
    "data/processed/transacciones_analiticas.csv"
)

CARPETA_PROCESSED = Path(
    "data/processed"
)

CARPETA_PROCESSED.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CARGA DE DATOS
# ============================================================

df_clientes = pd.read_csv(
    RUTA_CLIENTES
)

df_transacciones = pd.read_csv(
    RUTA_TRANSACCIONES,
    parse_dates=["fecha_hora"]
)


print("Dimensión de clientes:")
print(df_clientes.shape)

print("\nTabla de transacciones:")
print(df_transacciones.shape)


# ============================================================
# CONTROLES DE CALIDAD
# ============================================================

clientes_duplicados = (
    df_clientes["cliente_id"]
    .duplicated()
    .sum()
)

transacciones_duplicadas = (
    df_transacciones["transaccion_id"]
    .duplicated()
    .sum()
)

clientes_transacciones = set(
    df_transacciones["cliente_id"]
)

clientes_maestro = set(
    df_clientes["cliente_id"]
)

clientes_inexistentes = (
    clientes_transacciones -
    clientes_maestro
)

montos_invalidos = (
    df_transacciones["monto_soles"] <= 0
).sum()

fechas_fuera_periodo = (
    (
        df_transacciones["fecha_hora"] <
        "2025-01-01"
    )
    |
    (
        df_transacciones["fecha_hora"] >=
        "2026-01-01"
    )
).sum()


print("\nControles de calidad:")
print(
    "Clientes duplicados:",
    clientes_duplicados
)
print(
    "Transacciones duplicadas:",
    transacciones_duplicadas
)
print(
    "Clientes inexistentes:",
    len(clientes_inexistentes)
)
print(
    "Valores faltantes:",
    int(
        df_transacciones
        .isna()
        .sum()
        .sum()
    )
)
print(
    "Montos menores o iguales a cero:",
    montos_invalidos
)
print(
    "Fechas fuera de 2025:",
    fechas_fuera_periodo
)


# Detenemos el proceso si existen errores críticos
assert clientes_duplicados == 0
assert transacciones_duplicadas == 0
assert len(clientes_inexistentes) == 0
assert montos_invalidos == 0
assert fechas_fuera_periodo == 0

print(
    "\nTodos los controles críticos fueron superados."
)

# ============================================================
# CREACIÓN DE VARIABLES ANALÍTICAS
# ============================================================

# Creamos una fecha sin hora para relacionarla con el calendario
df_transacciones["fecha"] = (
    df_transacciones["fecha_hora"].dt.normalize()
)

# Indicadores para construir la matriz de confusión
df_transacciones["verdadero_positivo"] = (
    (
        (df_transacciones["alerta_generada"] == 1) &
        (df_transacciones["fraude_confirmado"] == 1)
    )
).astype(int)

df_transacciones["falso_positivo"] = (
    (
        (df_transacciones["alerta_generada"] == 1) &
        (df_transacciones["fraude_confirmado"] == 0)
    )
).astype(int)

df_transacciones["falso_negativo"] = (
    (
        (df_transacciones["alerta_generada"] == 0) &
        (df_transacciones["fraude_confirmado"] == 1)
    )
).astype(int)

df_transacciones["verdadero_negativo"] = (
    (
        (df_transacciones["alerta_generada"] == 0) &
        (df_transacciones["fraude_confirmado"] == 0)
    )
).astype(int)


# Segmentamos el importe de la operación
df_transacciones["rango_monto"] = pd.cut(
    df_transacciones["monto_soles"],
    bins=[
        0,
        100,
        500,
        1000,
        2000,
        np.inf
    ],
    labels=[
        "Hasta S/100",
        "S/101-S/500",
        "S/501-S/1,000",
        "S/1,001-S/2,000",
        "Más de S/2,000"
    ],
    include_lowest=True
)


# Clasificamos el puntaje generado por las reglas
df_transacciones["nivel_alerta"] = pd.cut(
    df_transacciones["puntaje_riesgo"],
    bins=[
        -1,
        39,
        54,
        69,
        100
    ],
    labels=[
        "Sin alerta",
        "Media",
        "Alta",
        "Crítica"
    ]
)


# ============================================================
# DIMENSIÓN FECHA
# ============================================================

df_fecha = pd.DataFrame({
    "fecha": pd.date_range(
        start=FECHA_INICIAL,
        end=FECHA_FINAL,
        freq="D"
    )
})

df_fecha["anio"] = (
    df_fecha["fecha"].dt.year
)

df_fecha["mes_numero"] = (
    df_fecha["fecha"].dt.month
)

meses_espanol = {
    1: "Enero",
    2: "Febrero",
    3: "Marzo",
    4: "Abril",
    5: "Mayo",
    6: "Junio",
    7: "Julio",
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre",
    12: "Diciembre"
}

df_fecha["mes_nombre"] = (
    df_fecha["mes_numero"]
    .map(meses_espanol)
)

df_fecha["anio_mes"] = (
    df_fecha["fecha"]
    .dt.strftime("%Y-%m")
)

df_fecha["trimestre"] = (
    "T" +
    df_fecha["fecha"]
    .dt.quarter
    .astype(str)
)

df_fecha["dia_semana_numero"] = (
    df_fecha["fecha"].dt.dayofweek + 1
)

df_fecha["es_fin_semana"] = (
    df_fecha["dia_semana_numero"] >= 6
).astype(int)


# ============================================================
# TABLA DE ALERTAS
# ============================================================

df_alertas = (
    df_transacciones[
        df_transacciones["alerta_generada"] == 1
    ]
    .copy()
)

df_alertas = (
    df_alertas
    .sort_values("fecha_hora")
    .reset_index(drop=True)
)

df_alertas.insert(
    0,
    "alerta_id",
    [
        f"ALT{i:07d}"
        for i in range(1, len(df_alertas) + 1)
    ]
)

columnas_alertas = [
    "alerta_id",
    "transaccion_id",
    "cliente_id",
    "fecha_hora",
    "puntaje_riesgo",
    "nivel_alerta",
    "fraude_confirmado",
    "resultado_monitoreo",
    "monto_soles",
    "monto_prevenido",
    "falso_positivo",
    "verdadero_positivo"
]

df_alertas = df_alertas[
    columnas_alertas
]


# ============================================================
# TABLA PARA EL MODELO DE CLASIFICACIÓN
# ============================================================

columnas_cliente_modelo = [
    "cliente_id",
    "edad",
    "segmento",
    "departamento",
    "antiguedad_meses",
    "ingreso_mensual",
    "nivel_riesgo_cliente",
    "canal_preferido"
]

df_modelo = df_transacciones.merge(
    df_clientes[columnas_cliente_modelo],
    on="cliente_id",
    how="left",
    suffixes=("", "_cliente")
)


# ============================================================
# GUARDADO DE TABLAS
# ============================================================

RUTA_DIM_CLIENTES = (
    CARPETA_PROCESSED /
    "dim_clientes.csv"
)

RUTA_DIM_FECHA = (
    CARPETA_PROCESSED /
    "dim_fecha.csv"
)

RUTA_FACT_TRANSACCIONES = (
    CARPETA_PROCESSED /
    "fact_transacciones.csv"
)

RUTA_FACT_ALERTAS = (
    CARPETA_PROCESSED /
    "fact_alertas.csv"
)

RUTA_DATASET_MODELO = (
    CARPETA_PROCESSED /
    "dataset_modelo_fraude.csv"
)


df_clientes.to_csv(
    RUTA_DIM_CLIENTES,
    index=False,
    encoding="utf-8-sig"
)

df_fecha.to_csv(
    RUTA_DIM_FECHA,
    index=False,
    encoding="utf-8-sig"
)

df_transacciones.to_csv(
    RUTA_FACT_TRANSACCIONES,
    index=False,
    encoding="utf-8-sig"
)

df_alertas.to_csv(
    RUTA_FACT_ALERTAS,
    index=False,
    encoding="utf-8-sig"
)

df_modelo.to_csv(
    RUTA_DATASET_MODELO,
    index=False,
    encoding="utf-8-sig"
)


print("\nTablas analíticas generadas:")
print("Dimensión clientes:", df_clientes.shape)
print("Dimensión fecha:", df_fecha.shape)
print("Hechos transacciones:", df_transacciones.shape)
print("Hechos alertas:", df_alertas.shape)
print("Dataset para modelo:", df_modelo.shape)

print("\nArchivos guardados correctamente.")

FECHA_INICIAL = "2025-01-01"
FECHA_FINAL = "2025-12-31"