"""
Maqueta: Sistema de Priorización de Camas Críticas (UCI/UTI)

Estima, a partir de datos conocidos al momento de decidir un traslado
(diagnóstico, edad, comorbilidades, procedimiento previsto, tipo de ingreso,
hospital de destino), la probabilidad de que un paciente requiera cama crítica
(UCI/UTI) durante su hospitalización.

Entrenado con datos públicos GRD de Chile (2019-2024, DEIS/MINSAL).

IMPORTANTE: esta es una herramienta de apoyo a la gestión basada en patrones
históricos de codificación clínica, NO un score clínico validado de
deterioro en tiempo real (como NEWS2). No reemplaza el juicio clínico.
"""

import datetime
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
MODELO_PATH = RAIZ / "models" / "modelo_riesgo_uci.joblib"

ESTILOS = """
<style>
  /* Streamlit deja mucho aire arriba y centra en un ancho estrecho; se recupera
     espacio útil y se fija un máximo cómodo de lectura. */
  .block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }
  #MainMenu, footer, header [data-testid="stStatusWidget"] { visibility: hidden; }

  /* Encabezado propio en vez del st.title genérico */
  .cabecera { border-bottom: 1px solid #E2E8F0; padding-bottom: 1.1rem; margin-bottom: 1.6rem; }
  .cabecera h1 { font-size: 1.85rem; font-weight: 650; letter-spacing: -0.02em;
                 color: #0F172A; margin: 0 0 .35rem 0; line-height: 1.2; }
  .cabecera p  { color: #64748B; font-size: .93rem; margin: 0; max-width: 62ch; line-height: 1.5; }
  .cabecera .etiqueta { display: inline-block; font-size: .7rem; font-weight: 600;
                        letter-spacing: .06em; text-transform: uppercase; color: #0E7490;
                        background: #ECFEFF; border: 1px solid #A5F3FC;
                        padding: .18rem .5rem; border-radius: 4px; margin-bottom: .6rem; }

  /* Pestañas con aspecto de navegación, no de botones sueltos */
  .stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid #E2E8F0; }
  .stTabs [data-baseweb="tab"] { padding: .55rem 0; font-weight: 500; }

  /* Tarjetas para los dos puntajes */
  .tarjeta { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px;
             padding: 1.1rem 1.3rem .5rem 1.3rem; height: 100%; }
  .tarjeta .titulo { font-size: 1.02rem; font-weight: 600; color: #0F172A; margin-bottom: .15rem; }
  .tarjeta .sub    { font-size: .82rem; color: #64748B; margin-bottom: .3rem; line-height: 1.45; }

  /* Secciones del formulario */
  .seccion { font-size: .74rem; font-weight: 650; letter-spacing: .07em;
             text-transform: uppercase; color: #94A3B8;
             margin: 1.7rem 0 .5rem 0; padding-bottom: .35rem;
             border-bottom: 1px solid #F1F5F9; }

  /* Casillas de comorbilidad más compactas: son 19 */
  .stCheckbox { margin-bottom: -.55rem; }
  .stCheckbox label p { font-size: .87rem; }

  /* El botón principal con más presencia */
  .stButton button[kind="primary"] { padding: .5rem 1.5rem; font-weight: 600; }

  div[data-testid="stMetricValue"] { font-size: 1.25rem; }
</style>
"""

st.set_page_config(
    page_title="Priorización de Camas Críticas UCI/UTI",
    page_icon="🏥",
    layout="wide",
)
st.markdown(ESTILOS, unsafe_allow_html=True)

NOMBRES_LEGIBLES = {
    "cod_hospital": "Hospital",
    "sexo": "Sexo",
    "prevision": "Previsión",
    "tipo_procedencia": "Procedencia",
    "tipo_ingreso": "Tipo de ingreso",
    "diagnostico1_categoria": "Diagnóstico principal",
    "procedimiento_principal": "Procedimiento principal",
    "tiene_mcc": "Tiene MCC",
    "tiene_cc": "Tiene CC",
    "nivel_severidad_potencial": "Nivel de severidad APR potencial",
    "tiene_diabetes": "Diabetes",
    "tiene_hipertension": "Hipertensión",
    "tiene_erc": "Enfermedad renal crónica",
    "tiene_epoc": "EPOC",
    "tiene_obesidad": "Obesidad",
    "diagnostico1_subcodigo": "Diagnóstico (código específico)",
    "tiene_insuf_cardiaca": "Insuficiencia cardíaca",
    "tiene_cardiopatia_isquemica": "Cardiopatía isquémica",
    "tiene_arritmia": "Arritmia",
    "tiene_acv_previo": "ACV previo",
    "tiene_vascular_periferica": "Enfermedad vascular periférica",
    "tiene_cancer": "Cáncer activo",
    "tiene_cancer_metastasico": "Cáncer metastásico",
    "tiene_hepatopatia": "Hepatopatía",
    "tiene_demencia": "Demencia",
    "tiene_inmunosupresion": "Inmunosupresión / VIH",
    "tiene_anemia": "Anemia",
    "tiene_desnutricion": "Desnutrición",
    "tiene_asma": "Asma",
    "tiene_tabaquismo": "Tabaquismo",
}
# agrupadas por sistema para que el formulario no sea una lista plana de 19 casillas
GRUPOS_COMORBILIDAD = {
    "Cardiovascular": [
        ("tiene_hipertension", "Hipertensión"),
        ("tiene_insuf_cardiaca", "Insuficiencia cardíaca"),
        ("tiene_cardiopatia_isquemica", "Cardiopatía isquémica"),
        ("tiene_arritmia", "Arritmia / fibrilación auricular"),
        ("tiene_acv_previo", "ACV previo"),
        ("tiene_vascular_periferica", "Enfermedad vascular periférica"),
    ],
    "Metabólico y renal": [
        ("tiene_diabetes", "Diabetes"),
        ("tiene_obesidad", "Obesidad"),
        ("tiene_erc", "Enfermedad renal crónica"),
        ("tiene_hepatopatia", "Hepatopatía / cirrosis"),
        ("tiene_desnutricion", "Desnutrición"),
    ],
    "Respiratorio": [
        ("tiene_epoc", "EPOC"),
        ("tiene_asma", "Asma"),
        ("tiene_tabaquismo", "Tabaquismo"),
    ],
    "Oncológico e inmune": [
        ("tiene_cancer", "Cáncer activo"),
        ("tiene_cancer_metastasico", "Cáncer metastásico"),
        ("tiene_inmunosupresion", "Inmunosupresión / VIH"),
        ("tiene_anemia", "Anemia"),
        ("tiene_demencia", "Demencia"),
    ],
}
COMORBILIDADES = [c for grupo in GRUPOS_COMORBILIDAD.values() for c, _ in grupo]


@st.cache_resource
def cargar_modelo():
    return joblib.load(MODELO_PATH)


def clasificar_riesgo(p, umbrales):
    if p >= umbrales["rojo"]:
        return "🔴 Alto", "#e74c3c"
    elif p >= umbrales["amarillo"]:
        return "🟡 Medio", "#f39c12"
    else:
        return "🟢 Bajo", "#2ecc71"


def gauge_riesgo(p, umbrales, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p * 100,
        number={"suffix": "%", "font": {"size": 34, "color": "#0F172A"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#CBD5E1", "tickfont": {"size": 10}},
            "bar": {"color": color, "thickness": 0.68},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, umbrales["amarillo"] * 100], "color": "rgba(46,204,113,0.25)"},
                {"range": [umbrales["amarillo"] * 100, umbrales["rojo"] * 100], "color": "rgba(243,156,18,0.25)"},
                {"range": [umbrales["rojo"] * 100, 100], "color": "rgba(231,76,60,0.25)"},
            ],
        },
    ))
    fig.update_layout(
        height=185, margin=dict(l=24, r=24, t=6, b=0),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="sans-serif", color="#475569"),
    )
    return fig


def construir_input(datos: dict, artefacto: dict) -> pd.DataFrame:
    fila = {col: datos[col] for col in artefacto["features_categoricas"] + artefacto["features_numericas"]}
    df = pd.DataFrame([fila])
    for col in artefacto["features_categoricas"]:
        categorias = artefacto["opciones"][col]
        df[col] = pd.Categorical(df[col].astype(str), categories=categorias)
    return df


MAX_FILAS_CSV = 5000

# rangos aceptables por variable numérica; None = sin tope
RANGOS_NUMERICOS = {
    "edad": (0, 110),
    "mes_ingreso": (1, 12),
    "tiene_mcc": (0, 1), "tiene_cc": (0, 1),
    "nivel_severidad_potencial": (1, 3),
    "tiene_diabetes": (0, 1), "tiene_hipertension": (0, 1),
    "tiene_erc": (0, 1), "tiene_epoc": (0, 1), "tiene_obesidad": (0, 1),
}


def ejemplo_plantilla(artefacto: dict) -> dict:
    """Fila de ejemplo con valores que el modelo sí reconoce, para la plantilla."""
    fila = {}
    for col in artefacto["features_categoricas"]:
        opts = artefacto["opciones"][col]
        preferidos = {"procedimiento_principal": "SIN_PROCEDIMIENTO"}
        fila[col] = preferidos.get(col) if preferidos.get(col) in opts else opts[0]
    for col in artefacto["features_numericas"]:
        lo, hi = RANGOS_NUMERICOS.get(col, (0, 0))
        fila[col] = {"edad": 65, "mes_ingreso": 1, "nivel_severidad_potencial": 1}.get(col, lo)
    fila["_id"] = "Ejemplo - reemplazar"
    return fila


def validar_csv(bruto: pd.DataFrame, artefacto: dict):
    """Devuelve (filas_validas, lista_de_problemas).

    Sin esto la app aceptaba su propia plantilla vacía y devolvía un riesgo de
    1.85% construido enteramente con valores faltantes, sin ninguna advertencia.
    """
    cat = artefacto["features_categoricas"]
    num = artefacto["features_numericas"]
    problemas = []

    faltantes = [c for c in cat + num if c not in bruto.columns]
    if faltantes:
        return None, [f"Faltan columnas obligatorias: {', '.join(faltantes)}"]

    df = bruto.copy()
    if len(df) > MAX_FILAS_CSV:
        problemas.append(f"El archivo trae {len(df):,} filas; se procesan las primeras {MAX_FILAS_CSV:,}.")
        df = df.head(MAX_FILAS_CSV)

    por_defecto = pd.Series([f"Fila {i + 1}" for i in range(len(df))], index=df.index)
    if "_id" not in df.columns:
        df["_id"] = por_defecto
    else:
        etiquetas = df["_id"].fillna("").astype(str).str.strip()
        df["_id"] = etiquetas.where(etiquetas != "", por_defecto)

    valida = pd.Series(True, index=df.index)

    for col in num:
        convertida = pd.to_numeric(df[col], errors="coerce")
        malos = convertida.isna()
        if malos.any():
            problemas.append(f"'{col}': {malos.sum()} fila(s) con valor vacío o no numérico.")
        lo, hi = RANGOS_NUMERICOS.get(col, (None, None))
        if lo is not None:
            fuera = convertida.notna() & ((convertida < lo) | (convertida > hi))
            if fuera.any():
                problemas.append(f"'{col}': {fuera.sum()} fila(s) fuera del rango {lo}-{hi}.")
            malos = malos | fuera
        df[col] = convertida
        valida &= ~malos

    for col in cat:
        permitidos = set(artefacto["opciones"][col])
        vals = df[col].fillna("").astype(str).str.strip()
        malos = ~vals.isin(permitidos)
        if malos.any():
            ejemplos = sorted(set(vals[malos]) - {""})[:3]
            detalle = f" (ej. {', '.join(repr(e) for e in ejemplos)})" if ejemplos else ""
            problemas.append(
                f"'{col}': {malos.sum()} fila(s) con un valor que el modelo no conoce{detalle}."
            )
        df[col] = vals
        valida &= ~malos

    descartadas = int((~valida).sum())
    if descartadas == len(df):
        problemas.append(f"Ninguna de las {len(df)} filas es utilizable.")
    elif descartadas:
        problemas.append(
            f"Se descartan {descartadas} de {len(df)} filas; las otras {len(df) - descartadas} sí se pueden usar."
        )
    return df[valida].copy(), problemas


def predecir_lote(casos: list[dict], artefacto: dict, modelo) -> "pd.Series":
    """Una sola llamada a predict_proba para toda la cola.

    Antes se llamaba fila por fila dentro de un bucle de Python, lo que hacía
    que una cola grande tardara segundos y bloqueara la app.
    """
    cols = artefacto["features_categoricas"] + artefacto["features_numericas"]
    X = pd.DataFrame([{c: caso.get(c) for c in cols} for caso in casos])
    for col in artefacto["features_categoricas"]:
        X[col] = pd.Categorical(X[col].astype(str), categories=artefacto["opciones"][col])
    for col in artefacto["features_numericas"]:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return modelo.predict_proba(X)[:, 1]


def panel_explicabilidad(datos: dict, artefacto: dict):
    tasas = artefacto["tasas_historicas"]
    base = artefacto["tasa_base_2024"]
    nombres_hosp = artefacto["opciones"]["cod_hospital_nombre"]
    filas = []
    for col in artefacto["features_categoricas"]:
        valor = str(datos[col])
        tasa_valor = tasas.get(col, {}).get(valor)
        if tasa_valor is None:
            continue
        diferencia = tasa_valor - base
        valor_legible = nombres_hosp.get(valor, valor) if col == "cod_hospital" else valor
        filas.append({
            "Factor": NOMBRES_LEGIBLES.get(col, col),
            "Valor": valor_legible,
            "Tasa histórica UCI/UTI": tasa_valor,
            "vs. promedio": diferencia,
        })
    for col in COMORBILIDADES + ["tiene_mcc", "tiene_cc"]:
        valor = int(datos.get(col, 0))
        tasa_valor = tasas.get(col, {}).get(valor)
        if tasa_valor is None or valor == 0:
            continue
        diferencia = tasa_valor - base
        filas.append({
            "Factor": NOMBRES_LEGIBLES.get(col, col),
            "Valor": "Sí",
            "Tasa histórica UCI/UTI": tasa_valor,
            "vs. promedio": diferencia,
        })
    if not filas:
        return
    tabla = pd.DataFrame(filas).sort_values("vs. promedio", ascending=False, key=abs)
    tabla["Tasa histórica UCI/UTI"] = (tabla["Tasa histórica UCI/UTI"] * 100).round(1).astype(str) + "%"
    tabla["vs. promedio"] = tabla["vs. promedio"].apply(lambda x: f"{'+' if x >= 0 else ''}{x*100:.1f} pp")
    st.caption(
        "Tasa histórica de necesidad de UCI/UTI observada en los datos de entrenamiento para cada "
        f"valor elegido, comparada con el promedio general ({base*100:.1f}%). No es la explicación "
        "exacta del modelo (que combina las variables de forma no lineal), pero da una idea de qué "
        "factores empujan el riesgo hacia arriba o abajo."
    )
    st.dataframe(tabla, width="stretch", hide_index=True)


# Valores de arranque del formulario. Sin esto la app abre con el primer valor
# alfabético de cada lista ("DESCONOCIDO", un hospital de Arica), que da una
# primera impresión pobre y no representa un caso típico.
PRECARGA = {
    "cod_hospital": "114101",  # Sótero del Río, el de mayor volumen del conjunto
    "sexo": "HOMBRE",
    "tipo_ingreso": "URGENCIA",
    "tipo_procedencia": "SERVICIO EMERGENCIA (DOMICILIO)",
    "prevision": "FONASA INSTITUCIONAL - (MAI) B",
    "diagnostico1_categoria": "J18",
    "procedimiento_principal": "SIN_PROCEDIMIENTO",
}


def indice_por_defecto(opciones: list, col: str) -> int:
    """Posición del valor de arranque, o 0 si ese valor no está disponible."""
    preferido = PRECARGA.get(col)
    return opciones.index(preferido) if preferido in opciones else 0


def selector_subcodigo(opciones: dict, categoria: str, key_prefix: str) -> str:
    """Segundo desplegable con el código CIE-10 específico dentro de la categoría.

    La categoría de 3 caracteres agrupa cuadros muy distintos: I21.0 (IAM anterior)
    tiene 76.9% de tasa de UCI/UTI y I21.4 (sin elevación del ST) un 52.9%. Se ofrece
    en cascada, filtrando solo los subcódigos que existen bajo la categoría elegida.
    """
    disponibles = opciones.get("subcodigos_por_categoria", {}).get(str(categoria), [])
    desc = opciones.get("subcodigo_descripcion", {})
    if not disponibles:
        return "OTRO"
    return st.selectbox(
        "Código específico",
        options=disponibles + ["OTRO"],
        format_func=lambda c: ("Otro / sin especificar" if c == "OTRO"
                               else f"{c} — {desc.get(c, '')}"[:70]),
        key=f"{key_prefix}_subcod",
        help="Detalle dentro de la categoría elegida. Discrimina bastante más que la categoría sola.",
    )


def formulario_paciente(artefacto: dict, key_prefix: str) -> dict:
    opciones = artefacto["opciones"]
    nombres_hosp = opciones["cod_hospital_nombre"]
    desc_diag = opciones["diagnostico1_descripcion"]

    st.markdown('<div class="seccion">Datos del paciente y del ingreso</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        hospital = st.selectbox(
            "Hospital",
            options=opciones["cod_hospital"],
            format_func=lambda c: f"{nombres_hosp.get(c, c)} ({c})",
            key=f"{key_prefix}_hosp",
            index=indice_por_defecto(opciones["cod_hospital"], "cod_hospital"),
        )
        sexo = st.selectbox("Sexo", options=opciones["sexo"], key=f"{key_prefix}_sexo",
            index=indice_por_defecto(opciones["sexo"], "sexo"))
        edad = st.number_input("Edad", min_value=0, max_value=110, value=60, key=f"{key_prefix}_edad")
    with col2:
        tipo_ingreso = st.selectbox(
            "Tipo de ingreso", options=opciones["tipo_ingreso"], key=f"{key_prefix}_tingreso",
            index=indice_por_defecto(opciones["tipo_ingreso"], "tipo_ingreso")
        )
        tipo_procedencia = st.selectbox(
            "Procedencia", options=opciones["tipo_procedencia"], key=f"{key_prefix}_tproc",
            index=indice_por_defecto(opciones["tipo_procedencia"], "tipo_procedencia")
        )
    with col3:
        prevision = st.selectbox("Previsión", options=opciones["prevision"], key=f"{key_prefix}_prev",
            index=indice_por_defecto(opciones["prevision"], "prevision"))
        diagnostico = st.selectbox(
            "Diagnóstico principal (CIE-10)",
            options=opciones["diagnostico1_categoria"],
            format_func=lambda c: f"{c} - {desc_diag.get(c, '')}",
            key=f"{key_prefix}_diag",
            index=indice_por_defecto(opciones["diagnostico1_categoria"], "diagnostico1_categoria"),
        )
        subcodigo = selector_subcodigo(opciones, diagnostico, key_prefix)
        mes = st.selectbox(
            "Mes de ingreso",
            options=list(range(1, 13)),
            index=datetime.date.today().month - 1,
            key=f"{key_prefix}_mes",
        )

    st.markdown('<div class="seccion">Procedimiento y comorbilidades</div>', unsafe_allow_html=True)
    st.caption(
        "El procedimiento se conoce exacto si es programado, o como intención si es de "
        "urgencia. Las comorbilidades se marcan según los diagnósticos ya conocidos."
    )
    procedimiento = st.selectbox(
        "Procedimiento principal previsto",
        options=opciones["procedimiento_principal"],
        format_func=lambda c: "Sin procedimiento" if c == "SIN_PROCEDIMIENTO" else (
            "Otro (fuera de los 200 más frecuentes)" if c == "OTRO" else c
        ),
        key=f"{key_prefix}_proc",
        index=indice_por_defecto(opciones["procedimiento_principal"], "procedimiento_principal"),
    )

    comorbilidades = {}
    columnas = st.columns(len(GRUPOS_COMORBILIDAD))
    for columna, (grupo, items) in zip(columnas, GRUPOS_COMORBILIDAD.items()):
        with columna:
            st.markdown(f"**{grupo}**")
            for campo, etiqueta in items:
                comorbilidades[campo] = int(st.checkbox(etiqueta, key=f"{key_prefix}_{campo}"))

    st.caption(
        "Severidad APR (según Módulo APR CC/MCC del IR-GRD chileno) — marca si algún "
        "diagnóstico secundario ya codificado corresponde a alguna de estas categorías:"
    )
    colm1, colm2 = st.columns(2)
    with colm1:
        tiene_mcc = st.checkbox(
            "Tiene MCC (complicación/comorbilidad MAYOR)", key=f"{key_prefix}_mcc",
            help="Ej: shock séptico, sepsis, insuficiencia respiratoria aguda, IRA, desnutrición moderada-severa.",
        )
    with colm2:
        tiene_cc = st.checkbox(
            "Tiene CC (complicación/comorbilidad, sin MCC)", key=f"{key_prefix}_cc",
            help="Ej: FA, EPOC exacerbado, ERC etapa 3-4, delirium, anemia activa.",
        )
    nivel_severidad_potencial = 3 if tiene_mcc else (2 if tiene_cc else 1)

    return {
        "cod_hospital": hospital,
        "sexo": sexo,
        "edad": edad,
        "mes_ingreso": mes,
        "prevision": prevision,
        "tipo_procedencia": tipo_procedencia,
        "tipo_ingreso": tipo_ingreso,
        "diagnostico1_categoria": diagnostico,
        "diagnostico1_subcodigo": subcodigo,
        "procedimiento_principal": procedimiento,
        "tiene_mcc": int(tiene_mcc),
        "tiene_cc": int(tiene_cc),
        "nivel_severidad_potencial": nivel_severidad_potencial,
        **comorbilidades,
    }


DIVERGENCIA_ALERTA_PP = 5.0  # puntos porcentuales


def tab_caso_individual(artefacto):
    cama, clinico = artefacto["cama_critica"], artefacto["riesgo_clinico"]
    datos = formulario_paciente(artefacto, "individual")
    if st.button("Calcular riesgo", type="primary"):
        X = construir_input(datos, artefacto)
        p_cama = cama["modelo"].predict_proba(X)[0, 1]
        p_clin = clinico["modelo"].predict_proba(X)[0, 1]
        et_cama, col_cama = clasificar_riesgo(p_cama, cama["umbrales_semaforo"])
        et_clin, col_clin = clasificar_riesgo(p_clin, clinico["umbrales_semaforo"])

        c1, c2 = st.columns(2)
        for columna, bloque, prob, etiqueta, color, clave, titulo, sub in [
            (c1, cama, p_cama, et_cama, col_cama, "g_cama", "Probabilidad de cama crítica",
             "Pregunta de <b>recurso</b>: ¿el destino tiene dónde ponerlo?"),
            (c2, clinico, p_clin, et_clin, col_clin, "g_clin", "Riesgo de desenlace adverso",
             "Pregunta <b>clínica</b>: ¿este traslado es seguro? (UCI/UTI o fallecimiento)"),
        ]:
            with columna:
                with st.container(border=True):
                    st.markdown(
                        f'<div class="titulo">{titulo}</div><div class="sub">{sub}</div>',
                        unsafe_allow_html=True,
                    )
                    st.plotly_chart(gauge_riesgo(prob, bloque["umbrales_semaforo"], color),
                                    width="stretch", key=clave)
                    st.markdown(f"### {etiqueta}")
                    st.caption(f"Tasa histórica de referencia: {bloque['tasa_base_2024']*100:.1f}%.")

        divergencia = (p_clin - p_cama) * 100
        if divergencia >= DIVERGENCIA_ALERTA_PP:
            st.warning(
                f"**Revisar con atención: divergencia de {divergencia:.1f} puntos.** El riesgo "
                f"clínico ({p_clin*100:.1f}%) supera bastante la probabilidad de cama crítica "
                f"({p_cama*100:.1f}%). A pacientes con este perfil históricamente se les asignó "
                f"cama crítica con menos frecuencia de lo que su desenlace sugeriría — es un patrón "
                f"típico en adultos mayores y pacientes frágiles. La decisión de traslado no "
                f"debería apoyarse solo en la disponibilidad de cama."
            )

        nivel_txt = {1: "1 — Sin CC/MCC", 2: "2 — Con CC", 3: "3 — Con MCC"}[datos["nivel_severidad_potencial"]]
        st.metric("Nivel de severidad APR potencial", nivel_txt)
        st.caption("Según reglas CC/MCC del Módulo APR (IR-GRD Chile) — misma lógica del agrupador real.")

        st.markdown("#### ¿Por qué este resultado?")
        panel_explicabilidad(datos, artefacto)


def tab_cola(artefacto):
    st.write("Agrega casos manualmente o sube un CSV con varios pacientes, y compáralos ordenados por riesgo.")
    if "cola" not in st.session_state:
        st.session_state.cola = []

    modo = st.radio("Modo de carga", ["Formulario manual", "Subir CSV"], horizontal=True, key="modo_carga")

    if modo == "Formulario manual":
        with st.form("agregar_caso"):
            datos_cola = formulario_paciente(artefacto, "cola")
            etiqueta_libre = st.text_input("Identificador del caso (opcional, ej. cama/box)", "")
            agregar = st.form_submit_button("➕ Agregar a la cola")
            if agregar:
                st.session_state.cola.append(
                    {**datos_cola, "_id": etiqueta_libre or f"Caso {len(st.session_state.cola)+1}"}
                )
    else:
        columnas_esperadas = artefacto["features_categoricas"] + artefacto["features_numericas"]
        st.caption("El CSV debe tener columnas: " + ", ".join(columnas_esperadas) + " (opcional: _id).")
        # La plantilla trae una fila de EJEMPLO con valores reales, no vacía: antes
        # se descargaba en blanco y al subirla sin tocar devolvía un riesgo de 1.85%
        # calculado sobre puros faltantes.
        plantilla = pd.DataFrame([ejemplo_plantilla(artefacto)])
        st.download_button(
            "⬇️ Descargar plantilla CSV",
            plantilla.to_csv(index=False).encode("utf-8"),
            file_name="plantilla_pacientes.csv",
            mime="text/csv",
        )
        archivo = st.file_uploader("Subir CSV de pacientes", type=["csv"])
        if archivo is not None:
            try:
                nuevo = pd.read_csv(archivo, dtype=str)
            except Exception as e:
                st.error(f"No se pudo leer el CSV: {e}")
                nuevo = None
            if nuevo is not None:
                limpio, problemas = validar_csv(nuevo, artefacto)
                if problemas:
                    st.error("El CSV tiene problemas:\n\n" + "\n".join(f"- {p}" for p in problemas))
                if limpio is not None and not limpio.empty:
                    st.success(f"{len(limpio)} filas válidas listas para agregar.")
                    st.dataframe(limpio.head(10), width="stretch", hide_index=True)
                    if st.button(f"➕ Agregar {len(limpio)} casos del CSV a la cola"):
                        st.session_state.cola.extend(limpio.to_dict(orient="records"))
                        st.rerun()

    if st.session_state.cola:
        cama, clinico = artefacto["cama_critica"], artefacto["riesgo_clinico"]
        p_cama = predecir_lote(st.session_state.cola, artefacto, cama["modelo"])
        p_clin = predecir_lote(st.session_state.cola, artefacto, clinico["modelo"])
        nombres_hosp = artefacto["opciones"]["cod_hospital_nombre"]
        filas = [
            {
                "Caso": caso["_id"],
                "Hospital": nombres_hosp.get(str(caso["cod_hospital"]), ""),
                "Edad": caso["edad"],
                "Diagnóstico": caso["diagnostico1_categoria"],
                "Procedimiento": caso["procedimiento_principal"],
                "Cama crítica": pc,
                "Riesgo clínico": pk,
                "Divergencia": (pk - pc) * 100,
                "Nivel clínico": clasificar_riesgo(pk, clinico["umbrales_semaforo"])[0],
            }
            for caso, pc, pk in zip(st.session_state.cola, p_cama, p_clin)
        ]
        # se ordena por riesgo CLÍNICO: es la pregunta que decide si el traslado es seguro
        tabla = pd.DataFrame(filas).sort_values("Riesgo clínico", ascending=False)
        n_divergentes = int((tabla["Divergencia"] >= DIVERGENCIA_ALERTA_PP).sum())
        for c in ("Cama crítica", "Riesgo clínico"):
            tabla[c] = (tabla[c] * 100).round(1).astype(str) + "%"
        tabla["Divergencia"] = tabla["Divergencia"].apply(
            lambda x: f"⚠️ +{x:.1f} pp" if x >= DIVERGENCIA_ALERTA_PP else f"{x:+.1f} pp"
        )
        st.caption("Ordenada por riesgo clínico. La columna *Divergencia* marca los casos donde el "
                   "riesgo clínico supera a la probabilidad de cama: revisar con atención.")
        st.dataframe(tabla, width="stretch", hide_index=True)
        if n_divergentes:
            st.warning(f"{n_divergentes} caso(s) con divergencia ≥ {DIVERGENCIA_ALERTA_PP:.0f} puntos.")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🗑️ Vaciar cola"):
                st.session_state.cola = []
                st.rerun()
        with col_b:
            st.download_button(
                "⬇️ Descargar cola como CSV",
                tabla.to_csv(index=False).encode("utf-8"),
                file_name="cola_priorizacion.csv",
                mime="text/csv",
            )
    else:
        st.info("Aún no has agregado casos a la cola.")


def tab_sobre_el_modelo(artefacto):
    st.markdown(f"""
    Entrenado con **{len(artefacto['opciones']['cod_hospital'])} hospitales** públicos chilenos con
    unidad crítica propia (datos GRD 2019-2023) y evaluado en **2024**, un año no visto durante
    el entrenamiento. Se excluyeron los hospitales sin unidad crítica propia, porque derivan a
    sus pacientes a otro centro y eso contaminaría la etiqueta.

    **Esta herramienta usa solo el código de diagnóstico y variables administrativas de ingreso
    — no reemplaza signos vitales ni el juicio clínico.** Es un prototipo para apoyar la
    priorización, no un dispositivo médico validado.
    """)

    st.markdown("#### Los dos modelos")
    # markdown en vez de st.dataframe: la tabla es de 2 filas y el dataframe
    # se renderizaba con las columnas colapsadas y las métricas invisibles.
    filas = ["| Modelo | ROC-AUC | Average Precision | Tasa base | AUC traslados | AP traslados |",
             "|---|---|---|---|---|---|"]
    for clave in ("cama_critica", "riesgo_clinico"):
        b = artefacto[clave]
        mt = b.get("metricas_traslados_2024") or {}
        filas.append(
            f"| {b['etiqueta']} | **{b['metricas_test_2024']['roc_auc']:.3f}** | "
            f"**{b['metricas_test_2024']['average_precision']:.3f}** | {b['tasa_base_2024']*100:.1f}% | "
            f"{mt.get('roc_auc', float('nan')):.3f} | {mt.get('average_precision', float('nan')):.3f} |"
        )
    st.markdown("\n".join(filas))

    with st.expander("⚠️ Por qué hacen falta DOS puntajes y no uno", expanded=True):
        st.markdown("""
        La etiqueta original mide **uso** de cama crítica, no necesidad clínica. Y el uso está
        sesgado por el racionamiento: a los pacientes mayores se les asigna menos UCI aunque
        estén más graves, por limitación del esfuerzo terapéutico y disponibilidad de camas.

        Medido en los datos, para **neumonía (J18)**:

        | edad | uso de UCI/UTI | mortalidad |
        |---|---|---|
        | menores de 50 | 22,3 % | 3,9 % |
        | 65 a 79 | 19,2 % | 17,6 % |
        | **80 o más** | **8,6 %** | **23,3 %** |

        El grupo de 80+ usa 2,6 veces menos UCI y muere 6 veces más. Mismo patrón en sepsis
        (55 % → 26 % de uso mientras la mortalidad sube de 18 % a 50 %).

        Un modelo entrenado solo con "¿ocupó cama crítica?" diría **riesgo bajo** justo para los
        pacientes más frágiles. Por eso se agregó el segundo puntaje, entrenado sobre
        **UCI/UTI o fallecimiento**, que no sufre ese sesgo del mismo modo.

        **Cuando los dos puntajes divergen, esa divergencia es la señal más útil**: riesgo
        clínico alto con probabilidad de cama baja marca exactamente los casos que el comité
        debe mirar dos veces.

        *Advertencia:* la mortalidad intrahospitalaria tampoco es un patrón oro puro — la
        limitación del esfuerzo terapéutico también influye en morir, no solo en recibir cama.
        Reduce mucho el sesgo, no lo elimina.
        """)

    with st.expander("¿Por qué el modelo no usa la especialidad médica ni el número de procedimientos?"):
        st.markdown("""
        Porque el GRD es una base de **egresos**: la ficha se llena al alta. Una columna solo
        sirve si su *valor* se conocía al momento de decidir, no solo si existe en el archivo.

        - `especialidad_medica` es el servicio tratante consolidado al alta. "Medicina Intensiva
          Adulto" tiene 98% de tasa de UCI/UTI: es prácticamente un sinónimo de la respuesta.
        - `n_diagnosticos_secundarios` y `n_procedimientos` cuentan casillas llenadas durante
          **toda** la estadía. Un paciente sale con 12 diagnósticos porque se complicó, no
          porque llegara así.

        Usarlas subía el ROC-AUC de 0.914 a 0.955, pero era una mejora que no existe en la vida
        real: al momento de decidir el traslado esos valores todavía no están.

        Tampoco se incluye la cirugía mayor ambulatoria (opera y se va el mismo día, 1.2% de
        tasa de UCI/UTI): no es objeto de una decisión de traslado y solo inflaba la métrica.
        """)

    COLORES = {"cama_critica": "#3498db", "riesgo_clinico": "#9b59b6"}

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Curva ROC (test 2024)")
        fig = go.Figure()
        for clave in ("cama_critica", "riesgo_clinico"):
            roc = artefacto[clave]["curva_roc"]
            fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines",
                                     name=artefacto[clave]["etiqueta"],
                                     line=dict(color=COLORES[clave], width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar",
                                  line=dict(color="gray", dash="dash")))
        fig.update_layout(
            xaxis_title="Falsos positivos", yaxis_title="Verdaderos positivos",
            height=350, margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(x=0.35, y=0.12, font=dict(size=10)),
        )
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("#### Calibración (predicho vs. real)")
        fig = go.Figure()
        for clave in ("cama_critica", "riesgo_clinico"):
            cal = artefacto[clave]["curva_calibracion"]
            fig.add_trace(go.Scatter(x=cal["mean_pred"], y=cal["frac_pos"], mode="lines+markers",
                                     name=artefacto[clave]["etiqueta"],
                                     line=dict(color=COLORES[clave], width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal",
                                  line=dict(color="gray", dash="dash")))
        fig.update_layout(
            xaxis_title="Probabilidad predicha", yaxis_title="Fracción real positiva",
            height=350, margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(x=0.35, y=0.12, font=dict(size=10)),
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### Importancia de variables")
    st.caption("Cuánto cae el ROC-AUC al mezclar aleatoriamente cada variable (permutation importance).")
    fig = go.Figure()
    orden = [f for f, _ in artefacto["cama_critica"]["importancia_variables"]]
    for clave in ("cama_critica", "riesgo_clinico"):
        imp = dict(artefacto[clave]["importancia_variables"])
        fig.add_trace(go.Bar(
            x=[imp.get(f, 0) for f in orden],
            y=[NOMBRES_LEGIBLES.get(f, f) for f in orden],
            orientation="h", name=artefacto[clave]["etiqueta"], marker_color=COLORES[clave],
        ))
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Caída en ROC-AUC", barmode="group",
                      legend=dict(x=0.5, y=0.08, font=dict(size=10)))
    st.plotly_chart(fig, width="stretch")


def main():
    artefacto = cargar_modelo()

    st.markdown(
        '<div class="cabecera">'
        '<span class="etiqueta">Prototipo · datos GRD públicos de Chile</span>'
        "<h1>Priorización de camas críticas</h1>"
        "<p>Estima el riesgo de un paciente al momento de decidir su traslado entre "
        "instituciones, a partir de patrones históricos de 5,8 millones de egresos "
        "hospitalarios. No reemplaza el juicio clínico.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["🧑‍⚕️ Evaluar caso individual", "📋 Cola de priorización", "📊 Sobre el modelo"])
    with tab1:
        tab_caso_individual(artefacto)
    with tab2:
        tab_cola(artefacto)
    with tab3:
        tab_sobre_el_modelo(artefacto)


if __name__ == "__main__":
    main()
