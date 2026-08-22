"""
Extrae features de ingreso y la etiqueta 'necesito_uci_uti' desde las bases
GRD públicas de Chile, leyendo directamente los .txt originales (no se copian
localmente) y guardando solo un parquet chico con los datos ya resumidos.

Los .txt crudos (~4 GB) se descargan del portal público del DEIS y NO se copian
al repositorio. Se indica dónde están con la variable de entorno GRD_DIR:

    export GRD_DIR="/ruta/a/bases de datos GRD"

Procesa un año a la vez y guarda cada uno en su propio parquet (data/por_anio/),
de modo que un año ya procesado no requiere que su .txt siga disponible. Esto
importa cuando el sistema de archivos evicta los originales por falta de espacio.

QUÉ ES FEATURE Y QUÉ NO
-----------------------
El GRD es una base de EGRESOS: la ficha entera se llena al alta. Una columna
solo puede ser feature si su VALOR habría sido conocible en el momento en que
el comité decide el traslado, no solo si existe en el archivo.

Se extraen igual, pero marcadas como auditoría (train_model.py NO las usa):
  - especialidad_medica  : es el servicio tratante consolidado al alta.
                           'MEDICINA INTENSIVA ADULTO' -> 98% de tasa UCI/UTI,
                           o sea casi un sinónimo de la etiqueta.
  - n_diagnosticos_secundarios / n_procedimientos
                         : cuentan casillas llenadas durante TODA la estadía,
                           no lo conocido al ingreso. Además n_procedimientos
                           queda truncado: se leen 10 slots pero el archivo
                           tiene 30, y el 26% de los casos supera los 10.
  - tipo_alta, severidad_grd, mortalidad_grd
                         : se calculan post-hoc, fuga evidente.

El índice de Charlson se eliminó: medido contra CC/MCC aportaba +0.001 de
Average Precision, dentro del ruido, y costaba ~35% del tiempo de extracción.
La carga de comorbilidad la llevan tiene_cc / tiene_mcc.
"""

import os
from pathlib import Path

import duckdb

PROYECTO = str(Path(__file__).resolve().parent.parent)
# Se usa la primera ruta donde el archivo esté realmente materializado.
RUTAS_BASE = [
    os.environ.get("GRD_DIR", ""),
    f"{PROYECTO}/bases de datos GRD",
]
DIR_POR_ANIO = f"{PROYECTO}/data/por_anio"
SALIDA = f"{PROYECTO}/data/admisiones.parquet"

# encoding detectado por archivo: el DEIS cambió de formato entre años
ARCHIVOS = [
    ("GRD_PUBLICO_2019.txt", "utf-8", 2019),
    ("GRD_PUBLICO_2020.txt", "utf-8", 2020),
    ("GRD_PUBLICO_2021.txt", "utf-8", 2021),
    ("GRD_PUBLICO_EXTERNO_2022.txt", "utf-16", 2022),
    ("GRD_PUBLICO_2023.txt", "utf-16", 2023),
    ("GRD_PUBLICO_2024.txt", "latin-1", 2024),
]

# servicios que cuentan como cama crítica (UCI/UTI, adulto/pediátrico/neonatal)
PATRONES_CRITICOS = [
    "CUIDADOS INTENSIVOS",
    "CUIDADOS INTERMEDIOS",
    "TRATAMIENTO INTERMEDIO",
    "TRATAMIENTOS INTERMEDIOS",
    "UTI ADULTOS",
    "(UCI)",
    "(UTI)",
]

TRASLADO_COLS = [f"SERVICIOTRASLADO{i}" for i in range(1, 10)]

# comorbilidades de alto riesgo a marcar como bandera, buscadas entre los
# diagnósticos secundarios (DIAGNOSTICO2..35) -- misma lógica que usan los
# sufijos W/CC (con complicación o comorbilidad) / W/MCC de la Norma GRD
COMORBILIDADES = {
    "tiene_diabetes": ["E10", "E11", "E12", "E13", "E14"],
    "tiene_hipertension": ["I10", "I11", "I12", "I13", "I15"],
    "tiene_erc": ["N18"],
    "tiene_epoc": ["J44"],
    "tiene_obesidad": ["E66"],
}

DIAG_SECUNDARIOS_COLS = [f"DIAGNOSTICO{i}" for i in range(2, 36)]
PROCEDIMIENTO_COLS = [f"PROCEDIMIENTO{i}" for i in range(1, 11)]

# Códigos CC (complicación/comorbilidad, severidad 2) y MCC (mayor, severidad
# 3-4) del sistema IR-GRD chileno real -- extraídos de
# NOTEBOOKLM_GRD_CHILE/documentos_unidos/modulo_APR_completo_CC_MCC.md
# (Tabla APR institucional + CMS MS-DRG Appendix C + IR-GRD v12 + MINSAL).
# Mucho más preciso para este contexto que un índice genérico como Charlson,
# porque es la clasificación que realmente usa el agrupador chileno.
MCC_PREFIJOS = [
    "A37", "A40", "A41.0", "A41.1", "A41.2", "A41.3", "A41.4", "A41.50", "A41.51", "A41.8", "A41.9",
    "B20", "B21", "B21.2", "B22", "B23", "B24", "B34", "C77", "C78", "C79", "C80",
    "D61.1", "D61.9", "D62", "D65", "D70",
    "E10.0", "E10.1", "E10.2", "E10.3", "E10.4", "E10.5", "E10.6", "E10.7",
    "E11.0", "E11.1", "E11.2", "E11.3", "E11.4", "E11.5", "E11.6", "E11.7",
    "E40", "E41", "E42", "E43", "E44.0", "E44.1", "E46",
    "G40", "G82", "G83.4", "G92", "G93.40", "G93.6",
    "I21", "I22", "I26.0", "I26.9", "I33.0", "I40", "I50.0", "I50.1", "I60", "I61", "I63",
    "J15", "J15.0", "J15.1", "J18", "J80", "J81", "J85", "J95.1", "J96.0", "J96.1", "J96.9",
    "K55.0", "K57", "K65.0", "K72.0", "K72.1", "K72.9", "K76.2", "K76.3", "K85",
    "L89.3", "L89.4", "N17.0", "N17.1", "N17.2", "N17.8", "N17.9", "N18.5",
    "R40.2", "R41.3", "R57.0", "R57.1", "R57.2", "R57.8", "R57.9",
    "S06", "T81", "Z99.2",
]
CC_PREFIJOS = [
    "C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09",
    "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19",
    "C20", "C21", "C22", "C23", "C24", "C25", "C26", "C27", "C28", "C29",
    "C30", "C31", "C32", "C33", "C34", "C35", "C36", "C37", "C38", "C39",
    "C40", "C41", "C42", "C43", "C44", "C45", "C46", "C47", "C48", "C49",
    "C50", "C51", "C52", "C53", "C54", "C55", "C56", "C57", "C58", "C59",
    "C60", "C61", "C62", "C63", "C64", "C65", "C66", "C67", "C68", "C69",
    "C70", "C71", "C72", "C73", "C74", "C75", "C76",
    "D37", "D38", "D39", "D40", "D41", "D42", "D43", "D44", "D45", "D46", "D47", "D48",
    "D50.0", "D63.0", "D64",
    "E10.9", "E11.9", "E24.2", "E56.8", "E74.8", "E86", "E87.0", "E87.1", "E87.2",
    "F05", "F10", "F20", "F32", "G30", "G35", "G55.1",
    "I20.0", "I25", "I27.2", "I42", "I44.2", "I48", "I48.9", "I50", "I80", "I82", "I95.9",
    "J44.1", "J45", "J46", "J84.9", "J90", "J98.1",
    "K50", "K51", "K57", "K71", "K74", "K80", "K86.1", "K92.1",
    "L03", "L89.1", "L89.2", "L97", "M05", "M79.3", "M86",
    "N04", "N13", "N18.3", "N18.4", "N28.1", "N39.0",
    "R04.2", "R13", "R14", "R47.0", "R47.1",
    "T81.0", "T81.1", "T81.2", "T81.3", "T81.4", "T81.8", "T82", "T83", "T84", "T85",
    "Y83", "Y84",
]

# procedimientos de soporte vital que ocurren DENTRO de la UCI/UTI (no se deciden
# antes del ingreso) -- usarlos como feature sería casi circular ("está ventilado"
# -> "necesitó UCI" es casi una tautología). Se excluyen de procedimiento_principal.
PROCEDIMIENTOS_SOPORTE_VITAL = [
    "96.70", "96.71", "96.72",  # ventilación mecánica invasiva
    "93.90",  # ventilación no invasiva
    "96.04",  # intubación endotraqueal
    "99.60", "99.61", "99.62", "99.63",  # reanimación cardiopulmonar
    "39.65",  # circulación extracorpórea (ECMO)
    "31.1",  # traqueostomía temporal (soporte de vía aérea en UCI)
]


def condicion_critico() -> str:
    campos = ["SERVICIOINGRESO"] + TRASLADO_COLS
    condiciones = []
    for campo in campos:
        for patron in PATRONES_CRITICOS:
            condiciones.append(f"{campo} ILIKE '%{patron}%'")
    return " OR ".join(condiciones)


def fecha(col: str) -> str:
    """Parsea fechas que vienen en YYYY-MM-DD (mayoría de años) o DD-MM-YYYY
    (2023 usa este formato) -- TRY_CAST sola falla en silencio con el segundo.
    Algunos años el CSV reader ya infiere DATE/TIMESTAMP automáticamente, por
    eso se castea a VARCHAR antes de intentar el parseo con formato explícito."""
    return (
        f"COALESCE("
        f"TRY_CAST({col} AS DATE), "
        f"TRY_STRPTIME({col}::VARCHAR, '%Y-%m-%d')::DATE, "
        f"TRY_STRPTIME({col}::VARCHAR, '%d-%m-%Y')::DATE"
        f")"
    )


def edad_sql() -> str:
    """Años CUMPLIDOS al ingreso.

    date_diff('year', ...) cuenta cruces de 1 de enero, no cumpleaños: un recién
    nacido del 31-dic ingresado el 1-ene daba edad=1. Importa mucho acá porque la
    tasa de UCI/UTI es 47% a los 0 años y 19% al año.
    """
    return f"date_part('year', age({fecha('FECHA_INGRESO')}, {fecha('FECHA_NACIMIENTO')}))"


def contar_no_nulos(columnas: list[str]) -> str:
    terminos = [f"(CASE WHEN {c} IS NOT NULL THEN 1 ELSE 0 END)" for c in columnas]
    return " + ".join(terminos)


def diagnosticos_concatenados() -> str:
    """Los 34 diagnósticos secundarios en un solo string separado por '|'.

    Permite buscar todos los prefijos de una familia con UN regex en vez de
    34 x N expresiones LIKE (medido: 13x más rápido). El COALESCE a '' es
    imprescindible: sin diagnósticos secundarios string_agg devuelve NULL y
    regexp_matches(NULL) es NULL, que contaminaría el resultado con nulos.
    """
    cols = ", ".join(DIAG_SECUNDARIOS_COLS)
    return (f"COALESCE(list_aggregate(list_filter([{cols}], x -> x IS NOT NULL), "
            f"'string_agg', '|'), '')")


def condicion_prefijos(prefijos: list[str]) -> str:
    """TRUE si algún diagnóstico secundario empieza con alguno de los prefijos."""
    patron = "|".join(sorted((p.replace(".", "\\.") for p in prefijos), key=len, reverse=True))
    return f"regexp_matches({diagnosticos_concatenados()}, '(^|\\|)({patron})')"


def main():
    os.makedirs(DIR_POR_ANIO, exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    condicion = condicion_critico()
    soporte_vital_sql = ", ".join(f"'{c}'" for c in PROCEDIMIENTOS_SOPORTE_VITAL)
    n_diag_sec_sql = contar_no_nulos(DIAG_SECUNDARIOS_COLS)
    n_proc_sql = contar_no_nulos(PROCEDIMIENTO_COLS)
    comorbilidad_cols_sql = ",\n                ".join(
        f"({condicion_prefijos(prefijos)})::INT AS {nombre}"
        for nombre, prefijos in COMORBILIDADES.items()
    )
    edad = edad_sql()

    for archivo, encoding, anio in ARCHIVOS:
        parquet_anio = f"{DIR_POR_ANIO}/{anio}.parquet"
        if os.path.exists(parquet_anio):
            print(f"{anio}: ya procesado ({parquet_anio}), se omite.")
            continue

        ruta = next((f"{b}/{archivo}" for b in RUTAS_BASE if os.path.exists(f"{b}/{archivo}")), None)
        if ruta is None:
            print(f"{anio}: {archivo} no está materializado en ninguna ruta (evictado a iCloud). Se omite por ahora.")
            continue

        print(f"Procesando {archivo} ({encoding})...")
        query = f"""
            WITH base AS (
                SELECT
                    *,
                    ({condicion_prefijos(MCC_PREFIJOS)}) AS _mcc,
                    ({condicion_prefijos(CC_PREFIJOS)}) AS _cc
                FROM read_csv(
                    '{ruta}',
                    delim='|', header=true, quote='', encoding='{encoding}',
                    sample_size=200000, ignore_errors=true
                )
                WHERE FECHA_NACIMIENTO IS NOT NULL AND FECHA_INGRESO IS NOT NULL
                  AND DIAGNOSTICO1 IS NOT NULL
                  AND {edad} BETWEEN 0 AND 110
            )
            SELECT
                {anio} AS anio_archivo,
                COD_HOSPITAL AS cod_hospital,
                SEXO AS sexo,
                {edad} AS edad,
                month({fecha('FECHA_INGRESO')}) AS mes_ingreso,
                PREVISION AS prevision,
                TIPO_PROCEDENCIA AS tipo_procedencia,
                TIPO_INGRESO AS tipo_ingreso,
                TIPO_ACTIVIDAD AS tipo_actividad,
                DIAGNOSTICO1 AS diagnostico1,
                SUBSTR(DIAGNOSTICO1, 1, 3) AS diagnostico1_categoria,
                _mcc::INT AS tiene_mcc,
                _cc::INT AS tiene_cc,
                CASE WHEN _mcc THEN 3 WHEN _cc THEN 2 ELSE 1 END AS nivel_severidad_potencial,
                {comorbilidad_cols_sql},
                CASE
                    WHEN PROCEDIMIENTO1 IN ({soporte_vital_sql}) THEN 'OTRO'
                    WHEN PROCEDIMIENTO1 IS NULL THEN 'SIN_PROCEDIMIENTO'
                    ELSE PROCEDIMIENTO1
                END AS procedimiento_principal,
                -- columnas SOLO DE AUDITORÍA: no se usan como features (ver docstring).
                ESPECIALIDAD_MEDICA AS especialidad_medica,
                ({n_diag_sec_sql}) AS n_diagnosticos_secundarios,
                ({n_proc_sql}) AS n_procedimientos,
                TIPOALTA AS tipo_alta,
                IR_29301_SEVERIDAD AS severidad_grd,
                IR_29301_MORTALIDAD AS mortalidad_grd,
                CASE WHEN ({condicion}) THEN 1 ELSE 0 END AS necesito_uci_uti
            FROM base
        """
        con.execute(f"COPY ({query}) TO '{parquet_anio}' (FORMAT PARQUET)")
        n = con.execute(f"SELECT count(*) FROM read_parquet('{parquet_anio}')").fetchone()[0]
        print(f"  {n:,} filas válidas -> {parquet_anio}")

    disponibles = sorted(
        f for f in os.listdir(DIR_POR_ANIO) if f.endswith(".parquet")
    )
    if not disponibles:
        print("No hay ningún año procesado todavía.")
        return

    anios_faltantes = {str(anio) for _, _, anio in ARCHIVOS} - {f.replace(".parquet", "") for f in disponibles}
    if anios_faltantes:
        print(f"\nAdvertencia: faltan por procesar los años {sorted(anios_faltantes)} "
              f"(archivo .txt no disponible localmente en este momento).")

    con.execute(
        f"CREATE OR REPLACE TABLE admisiones AS "
        f"SELECT * FROM read_parquet('{DIR_POR_ANIO}/*.parquet')"
    )
    total = con.execute("SELECT count(*) FROM admisiones").fetchone()[0]
    positivos = con.execute("SELECT sum(necesito_uci_uti) FROM admisiones").fetchone()[0]
    print(f"\nTotal admisiones combinadas ({len(disponibles)} años): {total:,}")
    print(f"Necesitaron UCI/UTI: {positivos:,} ({100*positivos/total:.1f}%)")

    con.execute(f"COPY admisiones TO '{SALIDA}' (FORMAT PARQUET)")
    print(f"\nGuardado en {SALIDA}")


if __name__ == "__main__":
    main()
