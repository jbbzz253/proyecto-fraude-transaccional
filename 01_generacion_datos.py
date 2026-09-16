from pathlib import Path
import numpy as np
import pandas as pd
from pathlib import Path

# Semilla para generar siempre los mismos datos ficticios
SEMILLA = 42
rng = np.random.default_rng(SEMILLA)


# Tamaño de las bases
NUMERO_CLIENTES = 5000
NUMERO_TRANSACCIONES = 100000


# Periodo simulado
FECHA_INICIAL = "2025-01-01"
FECHA_FINAL = "2025-12-31"


# Carpetas donde guardaremos los datos
CARPETA_RAW = Path("data/raw")
CARPETA_PROCESSED = Path("data/processed")

CARPETA_RAW.mkdir(parents=True, exist_ok=True)
CARPETA_PROCESSED.mkdir(parents=True, exist_ok=True)


print("Configuración del proyecto:")
print("Clientes:", NUMERO_CLIENTES)
print("Transacciones:", NUMERO_TRANSACCIONES)
print("Periodo:", FECHA_INICIAL, "a", FECHA_FINAL)

# ============================================================
# GENERACIÓN DE CLIENTES FICTICIOS
# ============================================================

# Creamos un identificador único para cada cliente
clientes_id = [
    f"CLI{i:06d}"
    for i in range(1, NUMERO_CLIENTES + 1)
]


# Asignamos un segmento comercial
segmentos = rng.choice(
    ["Masivo", "Preferente", "Premium"],
    size=NUMERO_CLIENTES,
    p=[0.70, 0.23, 0.07]
)


# Generamos edades entre 18 y 75 años
edades = rng.normal(
    loc=38,
    scale=12,
    size=NUMERO_CLIENTES
)

edades = np.clip(
    edades,
    18,
    75
).round().astype(int)


# Asignamos una ubicación ficticia
departamentos = rng.choice(
    [
        "Lima",
        "Callao",
        "Arequipa",
        "La Libertad",
        "Piura",
        "Cusco",
        "Junín",
        "Otros"
    ],
    size=NUMERO_CLIENTES,
    p=[
        0.48,
        0.10,
        0.08,
        0.08,
        0.07,
        0.06,
        0.05,
        0.08
    ]
)


# Generamos la antigüedad del cliente en meses
antiguedad_meses = rng.integers(
    low=1,
    high=121,
    size=NUMERO_CLIENTES
)


# Definimos un ingreso base según el segmento
ingreso_base = np.select(
    [
        segmentos == "Masivo",
        segmentos == "Preferente",
        segmentos == "Premium"
    ],
    [
        1800,
        4500,
        9000
    ]
)


# Agregamos variación individual a los ingresos
ingresos_mensuales = (
    ingreso_base *
    rng.lognormal(
        mean=0,
        sigma=0.35,
        size=NUMERO_CLIENTES
    )
).round(2)


# Asignamos un nivel inicial de riesgo
niveles_riesgo = rng.choice(
    ["Bajo", "Medio", "Alto"],
    size=NUMERO_CLIENTES,
    p=[0.72, 0.23, 0.05]
)


# Asignamos el canal utilizado con mayor frecuencia
canales_preferidos = rng.choice(
    ["Aplicación móvil", "Web", "POS", "Cajero"],
    size=NUMERO_CLIENTES,
    p=[0.52, 0.18, 0.22, 0.08]
)


# Construimos la tabla de clientes
df_clientes = pd.DataFrame({
    "cliente_id": clientes_id,
    "edad": edades,
    "segmento": segmentos,
    "departamento": departamentos,
    "antiguedad_meses": antiguedad_meses,
    "ingreso_mensual": ingresos_mensuales,
    "nivel_riesgo_cliente": niveles_riesgo,
    "canal_preferido": canales_preferidos
})


# Guardamos la dimensión de clientes
RUTA_CLIENTES = (
    CARPETA_RAW /
    "clientes_ficticios.csv"
)

df_clientes.to_csv(
    RUTA_CLIENTES,
    index=False,
    encoding="utf-8-sig"
)


# Mostramos controles básicos
print("\nPrimeros cinco clientes:")
print(df_clientes.head())

print("\nDimensiones de clientes:")
print(df_clientes.shape)

print("\nDistribución por nivel de riesgo:")
print(
    df_clientes[
        "nivel_riesgo_cliente"
    ].value_counts()
)

print("\nClientes guardados en:")
print(RUTA_CLIENTES)

# ============================================================
# GENERACIÓN DE TRANSACCIONES FICTICIAS
# ============================================================

# Algunos clientes realizan más operaciones que otros
pesos_actividad = rng.gamma(
    shape=2,
    scale=1,
    size=NUMERO_CLIENTES
)

pesos_actividad = (
    pesos_actividad /
    pesos_actividad.sum()
)


# Seleccionamos al cliente de cada transacción
clientes_transaccion = rng.choice(
    df_clientes["cliente_id"].to_numpy(),
    size=NUMERO_TRANSACCIONES,
    p=pesos_actividad
)


# Creamos identificadores únicos
transacciones_id = [
    f"TXN{i:08d}"
    for i in range(1, NUMERO_TRANSACCIONES + 1)
]


# Generamos fechas y horas durante todo 2025
fecha_inicio = pd.Timestamp(FECHA_INICIAL)

fecha_fin = (
    pd.Timestamp(FECHA_FINAL) +
    pd.Timedelta(days=1)
)

segundos_periodo = int(
    (fecha_fin - fecha_inicio).total_seconds()
)

segundos_aleatorios = rng.integers(
    low=0,
    high=segundos_periodo,
    size=NUMERO_TRANSACCIONES
)

fechas_horas = (
    fecha_inicio +
    pd.to_timedelta(
        segundos_aleatorios,
        unit="s"
    )
)


# Asignamos el canal de la operación
canales = rng.choice(
    ["Aplicación móvil", "Web", "POS", "Cajero"],
    size=NUMERO_TRANSACCIONES,
    p=[0.42, 0.16, 0.31, 0.11]
)


# Asignamos el tipo de operación
tipos_operacion = rng.choice(
    [
        "Compra",
        "Transferencia",
        "Retiro",
        "Pago de servicios"
    ],
    size=NUMERO_TRANSACCIONES,
    p=[0.52, 0.25, 0.12, 0.11]
)


# Monto típico según el tipo de operación
monto_base = np.select(
    [
        tipos_operacion == "Compra",
        tipos_operacion == "Transferencia",
        tipos_operacion == "Retiro",
        tipos_operacion == "Pago de servicios"
    ],
    [
        80,
        450,
        300,
        120
    ]
)


# Generamos montos positivos con algunos valores elevados
montos = (
    monto_base *
    rng.lognormal(
        mean=0,
        sigma=0.90,
        size=NUMERO_TRANSACCIONES
    )
)

montos = np.clip(
    montos,
    2,
    20000
).round(2)


# Categoría del establecimiento o destino
categorias = rng.choice(
    [
        "Supermercados",
        "Restaurantes",
        "Tecnología",
        "Entretenimiento",
        "Transporte",
        "Servicios",
        "Transferencias",
        "Retiros"
    ],
    size=NUMERO_TRANSACCIONES,
    p=[0.18, 0.15, 0.10, 0.10, 0.12, 0.12, 0.15, 0.08]
)


# Indicadores asociados con posibles anomalías
dispositivo_nuevo = rng.choice(
    [0, 1],
    size=NUMERO_TRANSACCIONES,
    p=[0.93, 0.07]
)

operacion_internacional = rng.choice(
    [0, 1],
    size=NUMERO_TRANSACCIONES,
    p=[0.95, 0.05]
)

intentos_fallidos_24h = rng.poisson(
    lam=0.30,
    size=NUMERO_TRANSACCIONES
)

intentos_fallidos_24h = np.clip(
    intentos_fallidos_24h,
    0,
    6
)

transacciones_ultima_hora = (
    rng.poisson(
        lam=1.1,
        size=NUMERO_TRANSACCIONES
    ) + 1
)

transacciones_ultima_hora = np.clip(
    transacciones_ultima_hora,
    1,
    12
)


# Distancia respecto a la ubicación habitual del cliente
distancia_ubicacion_km = rng.exponential(
    scale=18,
    size=NUMERO_TRANSACCIONES
)

# Algunas operaciones ocurren muy lejos de la ubicación habitual
ubicacion_atipica = (
    rng.random(NUMERO_TRANSACCIONES) < 0.04
)

distancia_ubicacion_km[ubicacion_atipica] += (
    rng.uniform(
        200,
        1500,
        size=ubicacion_atipica.sum()
    )
)

distancia_ubicacion_km = (
    distancia_ubicacion_km.round(2)
)


# Construimos la tabla de transacciones
df_transacciones = pd.DataFrame({
    "transaccion_id": transacciones_id,
    "cliente_id": clientes_transaccion,
    "fecha_hora": fechas_horas,
    "canal": canales,
    "tipo_operacion": tipos_operacion,
    "categoria": categorias,
    "monto_soles": montos,
    "dispositivo_nuevo": dispositivo_nuevo,
    "operacion_internacional": operacion_internacional,
    "intentos_fallidos_24h": intentos_fallidos_24h,
    "transacciones_ultima_hora": transacciones_ultima_hora,
    "distancia_ubicacion_km": distancia_ubicacion_km
})


# Extraemos variables de tiempo
df_transacciones["hora"] = (
    df_transacciones["fecha_hora"].dt.hour
)

df_transacciones["dia_semana"] = (
    df_transacciones["fecha_hora"].dt.day_name()
)

df_transacciones["mes"] = (
    df_transacciones["fecha_hora"].dt.month
)

df_transacciones["horario_nocturno"] = (
    (df_transacciones["hora"] < 6) |
    (df_transacciones["hora"] >= 23)
).astype(int)


# Ordenamos cronológicamente
df_transacciones = (
    df_transacciones
    .sort_values("fecha_hora")
    .reset_index(drop=True)
)


# Guardamos la base sin etiqueta de fraude
RUTA_TRANSACCIONES = (
    CARPETA_RAW /
    "transacciones_ficticias.csv"
)

df_transacciones.to_csv(
    RUTA_TRANSACCIONES,
    index=False,
    encoding="utf-8-sig"
)


# Controles básicos
print("\nPrimeras cinco transacciones:")
print(df_transacciones.head())

print("\nDimensiones de transacciones:")
print(df_transacciones.shape)

print("\nMonto promedio:")
print(round(df_transacciones["monto_soles"].mean(), 2))

print("\nTransacciones guardadas en:")
print(RUTA_TRANSACCIONES)

# ============================================================
# GENERACIÓN DEL FRAUDE Y LAS ALERTAS
# ============================================================

# Incorporamos el riesgo del cliente a cada transacción
mapa_riesgo = (
    df_clientes
    .set_index("cliente_id")[
        "nivel_riesgo_cliente"
    ]
)

df_transacciones["nivel_riesgo_cliente"] = (
    df_transacciones["cliente_id"]
    .map(mapa_riesgo)
)


# Construimos una función de riesgo no observable
riesgo_logit = (
    -5.20
    + 0.85 * df_transacciones["dispositivo_nuevo"]
    + 1.00 * df_transacciones["operacion_internacional"]
    + 0.75 * df_transacciones["horario_nocturno"]
    + 0.80 * (
        df_transacciones["intentos_fallidos_24h"] >= 2
    ).astype(int)
    + 0.95 * (
        df_transacciones["transacciones_ultima_hora"] >= 5
    ).astype(int)
    + 1.10 * (
        df_transacciones["distancia_ubicacion_km"] >= 200
    ).astype(int)
    + 0.65 * (
        df_transacciones["monto_soles"] >= 2000
    ).astype(int)
    + 0.70 * (
        df_transacciones["nivel_riesgo_cliente"] == "Alto"
    ).astype(int)
    + 0.25 * (
        df_transacciones["nivel_riesgo_cliente"] == "Medio"
    ).astype(int)
    + 0.30 * (
        df_transacciones["canal"] == "Web"
    ).astype(int)
)


# Convertimos el puntaje logit en una probabilidad entre 0 y 1
probabilidad_fraude = (
    1 /
    (1 + np.exp(-riesgo_logit))
)

df_transacciones["probabilidad_fraude"] = (
    probabilidad_fraude.round(6)
)


# Generamos la etiqueta real de fraude
df_transacciones["fraude_confirmado"] = (
    rng.random(NUMERO_TRANSACCIONES) <
    probabilidad_fraude
).astype(int)


# Creamos un puntaje de reglas entre 0 y 100
df_transacciones["puntaje_riesgo"] = (
    20 * df_transacciones["dispositivo_nuevo"]
    + 20 * df_transacciones["operacion_internacional"]
    + 15 * df_transacciones["horario_nocturno"]
    + 20 * (
        df_transacciones["intentos_fallidos_24h"] >= 2
    ).astype(int)
    + 20 * (
        df_transacciones["transacciones_ultima_hora"] >= 5
    ).astype(int)
    + 25 * (
        df_transacciones["distancia_ubicacion_km"] >= 200
    ).astype(int)
    + 15 * (
        df_transacciones["monto_soles"] >= 2000
    ).astype(int)
    + 10 * (
        df_transacciones["nivel_riesgo_cliente"] == "Alto"
    ).astype(int)
)

df_transacciones["puntaje_riesgo"] = (
    df_transacciones["puntaje_riesgo"]
    .clip(upper=100)
)


# Una alerta se genera desde 40 puntos
df_transacciones["alerta_generada"] = (
    df_transacciones["puntaje_riesgo"] >= 40
).astype(int)


# Clasificamos el resultado del sistema
condiciones_resultado = [
    (
        (df_transacciones["alerta_generada"] == 1) &
        (df_transacciones["fraude_confirmado"] == 1)
    ),
    (
        (df_transacciones["alerta_generada"] == 1) &
        (df_transacciones["fraude_confirmado"] == 0)
    ),
    (
        (df_transacciones["alerta_generada"] == 0) &
        (df_transacciones["fraude_confirmado"] == 1)
    )
]

resultados = [
    "Fraude detectado",
    "Falso positivo",
    "Fraude no detectado"
]

df_transacciones["resultado_monitoreo"] = np.select(
    condiciones_resultado,
    resultados,
    default="Operación normal"
)


# Calculamos impacto económico
df_transacciones["monto_prevenido"] = np.where(
    (
        (df_transacciones["fraude_confirmado"] == 1) &
        (df_transacciones["alerta_generada"] == 1)
    ),
    df_transacciones["monto_soles"],
    0
)

df_transacciones["perdida_por_fraude"] = np.where(
    (
        (df_transacciones["fraude_confirmado"] == 1) &
        (df_transacciones["alerta_generada"] == 0)
    ),
    df_transacciones["monto_soles"],
    0
)


# Guardamos la base analítica
RUTA_ANALITICA = (
    CARPETA_PROCESSED /
    "transacciones_analiticas.csv"
)

df_transacciones.to_csv(
    RUTA_ANALITICA,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# INDICADORES DE CONTROL
# ============================================================

total_fraudes = (
    df_transacciones["fraude_confirmado"].sum()
)

total_alertas = (
    df_transacciones["alerta_generada"].sum()
)

fraudes_detectados = (
    (
        (df_transacciones["fraude_confirmado"] == 1) &
        (df_transacciones["alerta_generada"] == 1)
    )
    .sum()
)

falsos_positivos = (
    (
        (df_transacciones["fraude_confirmado"] == 0) &
        (df_transacciones["alerta_generada"] == 1)
    )
    .sum()
)

precision_alertas = (
    fraudes_detectados /
    total_alertas
)

tasa_deteccion = (
    fraudes_detectados /
    total_fraudes
)


print("\nIndicadores generados:")
print("Total de transacciones:", len(df_transacciones))
print("Fraudes confirmados:", total_fraudes)
print("Alertas generadas:", total_alertas)
print("Fraudes detectados:", fraudes_detectados)
print("Falsos positivos:", falsos_positivos)

print(
    "Tasa de fraude:",
    round(
        100 * total_fraudes / len(df_transacciones),
        2
    ),
    "%"
)

print(
    "Precisión de alertas:",
    round(100 * precision_alertas, 2),
    "%"
)

print(
    "Tasa de detección:",
    round(100 * tasa_deteccion, 2),
    "%"
)

print(
    "Monto prevenido: S/",
    round(
        df_transacciones["monto_prevenido"].sum(),
        2
    )
)

print(
    "Pérdida por fraude: S/",
    round(
        df_transacciones["perdida_por_fraude"].sum(),
        2
    )
)

print("\nBase analítica guardada en:")
print(RUTA_ANALITICA)
