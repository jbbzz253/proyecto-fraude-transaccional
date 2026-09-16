-- ==========================================================
-- DESEMPEÑO DEL SISTEMA DE FRAUDE POR CANAL
-- ==========================================================

WITH resumen_canal AS (
    SELECT
        canal,
        COUNT(*) AS total_transacciones,
        ROUND(SUM(monto_soles), 2) AS monto_transaccionado,
        SUM(fraude_confirmado) AS fraudes_confirmados,
        SUM(alerta_generada) AS alertas_generadas,
        SUM(verdadero_positivo) AS fraudes_detectados,
        SUM(falso_positivo) AS falsos_positivos,
        SUM(falso_negativo) AS fraudes_no_detectados,
        ROUND(SUM(monto_prevenido), 2) AS monto_prevenido,
        ROUND(SUM(perdida_por_fraude), 2) AS perdida_por_fraude
    FROM fraude.fact_transacciones
    GROUP BY canal
)

SELECT
    canal,
    total_transacciones,
    monto_transaccionado,
    fraudes_confirmados,
    alertas_generadas,
    fraudes_detectados,
    falsos_positivos,
    fraudes_no_detectados,

    ROUND(
        100.0 * fraudes_confirmados /
        NULLIF(total_transacciones, 0),
        2
    ) AS tasa_fraude_pct,

    ROUND(
        100.0 * fraudes_detectados /
        NULLIF(alertas_generadas, 0),
        2
    ) AS precision_alertas_pct,

    ROUND(
        100.0 * fraudes_detectados /
        NULLIF(fraudes_confirmados, 0),
        2
    ) AS tasa_deteccion_pct,

    ROUND(
        100.0 * falsos_positivos /
        NULLIF(alertas_generadas, 0),
        2
    ) AS alertas_falsas_pct,

    monto_prevenido,
    perdida_por_fraude

FROM resumen_canal
ORDER BY perdida_por_fraude DESC;
