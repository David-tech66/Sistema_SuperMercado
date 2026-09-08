"""
Sistema_SuperMercado.py
========================
Interfaz web con Streamlit que integra:
  - Dashboard con indicadores y gráficos estadísticos
  - Pantalla de Predicción de ventas (ml_ventas.py)
  - Pantalla de Análisis de Opiniones (lógica de nlp_opiniones.py)
  - Pantalla de Asistente de Decisiones con recomendaciones automáticas
"""

import os
import sys
import warnings
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# STOPWORDS en español (sin necesidad de nltk)
# ──────────────────────────────────────────────────────────────────────────────

STOPWORDS_ES = {
    "a", "al", "algo", "algos", "alguna", "algunas", "alguno", "algunos", "aunque",
    "ante", "antes", "como", "con", "contra", "cual", "cuando", "de", "del", "desde",
    "donde", "durante", "e", "el", "ella", "ellas", "ellos", "en", "entre", "era",
    "erais", "eran", "eras", "eres", "es", "esa", "esas", "ese", "eso", "esos",
    "esta", "estas", "este", "esto", "estos", "fue", "fueron", "fui", "fuimos",
    "ha", "han", "has", "hasta", "hay", "he", "hemos", "her", "his", "hizo", "hubiera",
    "i", "igual", "la", "las", "le", "les", "lo", "los", "me", "mi", "mis", "mucho",
    "muchos", "muy", "más", "más", "o", "os", "otra", "otras", "otro", "otros",
    "para", "pero", "por", "porque", "que", "quien", "quienes", "se", "sea",
    "seas", "ser", "si", "sin", "sobre", "son", "su", "sus", "también", "tanto",
    "te", "ti", "tiene", "tienen", "todo", "todos", "tu", "tus", "u", "un", "una",
    "unas", "uno", "unos", "usted", "ustedes", "vez", "y", "ya", "yo",
}
# NOTA: "no" se excluye intencionalmente porque cambia el sentimiento

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE RUTAS
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.join(BASE_DIR, "modules")
CSV_LIMPIO = os.path.join(BASE_DIR, "supermercado_limpio.csv")

# Agregar modules/ al path para importar dashboard y ml_ventas
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTAR MÓDULOS (solo los que son importables sin efectos secundarios)
# ──────────────────────────────────────────────────────────────────────────────

import ml_ventas

from dashboard import (
    grafico_ventas_por_mes,
    grafico_productos_mas_vendidos,
    grafico_ventas_por_categoria,
    grafico_distribucion_precios,
    grafico_correlacion,
    grafico_histograma_satisfaccion,
)



# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN STREAMLIT
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sistema SuperMercado IA",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado
st.markdown("""
<style>
/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a3a5c 0%, #0d2137 100%);
}
[data-testid="stSidebar"] * { color: #e8f4fd !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 15px !important; }

/* Métricas */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e0e8f0;
    border-radius: 10px;
    padding: 12px 18px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.07);
}

/* Título principal */
h1 { color: #1a3a5c !important; }
h2 { color: #1a3a5c !important; border-bottom: 2px solid #4a90d9; padding-bottom: 6px; }

/* Recuadro recomendaciones */
.reco-box {
    background: linear-gradient(135deg, #e8f4fd, #f0f8e8);
    border-left: 5px solid #2980b9;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
    font-size: 15px;
    line-height: 1.7;
}
.reco-title {
    font-size: 18px;
    font-weight: 700;
    color: #1a3a5c;
    margin-bottom: 10px;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# CARGA Y CACHÉ DE DATOS
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Cargando datos…")
def cargar_df() -> pd.DataFrame:
    """Carga supermercado_limpio.csv con rutas absolutas."""
    if not os.path.exists(CSV_LIMPIO):
        st.error(
            "No se encontró **supermercado_limpio.csv**. "
            "Ejecuta primero `supermercado2.py` para generarlo."
        )
        st.stop()
    df = pd.read_csv(CSV_LIMPIO, encoding="utf-8-sig")
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    return df

# ──────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO ML (cacheado para no re-entrenar en cada interacción)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Entrenando modelos ML…")
def entrenar_ml():
    """
    Entrena los modelos de ml_ventas.py y devuelve el DataFrame
    de métricas y el DataFrame de referencia.
    """
    df = cargar_df()
    tabla = ml_ventas.entrenar_y_comparar(df, verbose=False)
    return tabla, df

# ──────────────────────────────────────────────────────────────────────────────
# NLP — MODELO DE OPINIONES
# Replica la lógica de nlp_opiniones.py sin ejecutar su código top-level
# ──────────────────────────────────────────────────────────────────────────────

def _limpiar_texto_nlp(texto: str) -> str:
    """Limpia una opinión: minúsculas, sin símbolos, sin stopwords (excepto 'no')."""
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúñü\s]", "", texto)
    palabras = texto.split()
    palabras = [p for p in palabras if p not in STOPWORDS_ES]
    return " ".join(palabras)


@st.cache_resource(show_spinner="Entrenando modelo NLP…")
def entrenar_nlp():
    """
    Entrena el modelo NLP de opiniones usando las listas de frases
    de nlp_opiniones.py más los ejemplos adicionales.
    Devuelve (vectorizador, modelo, precisión).
    """
    # Frases base (idénticas a nlp_opiniones.py)
    positivos_base = [
        "buenos precios y productos frescos",
        "la compra fue rápida y ordenada",
        "ofertas interesantes y personal amable",
        "muy buena variedad de productos",
        "excelente atención y encontré todo lo que necesitaba",
        "excelente!!!",
        "todo bien",
    ]
    negativos_base = [
        "había productos vencidos en el estante",
        "faltaban varios productos básicos",
        "los precios no coincidían con la etiqueta",
        "mala atención del personal",
        "mucha cola para pagar",
    ]

    # Ejemplos adicionales (idénticos a nlp_opiniones.py)
    positivas_extra = [
        "recomiendo este supermercado", "los recomiendo",
        "recomiendo comprar aquí", "muy buena experiencia",
        "la atención fue excelente", "excelente atención del personal",
        "hay muchos productos disponibles", "hay suficientes productos",
        "los productos están frescos", "los precios son buenos",
        "la compra fue rápida", "encontré todo lo que buscaba",
        "encontré todo lo que necesitaba", "volvería a comprar aquí",
        "me gustó la atención", "estoy satisfecho con la compra",
        "me gustó mucho la variedad", "todo estuvo muy bien",
        "la experiencia fue muy buena", "compraría nuevamente aquí",
        "me gustó mucho comprar aquí", "la atención fue muy buena",
        "los productos tienen buena calidad", "encontré buenos productos",
        "los precios son económicos", "la compra fue excelente",
        "volvería nuevamente", "estoy muy satisfecho",
        "recomiendo totalmente este lugar", "todo estuvo excelente",
        "la variedad es muy buena", "me atendieron muy bien",
        "encontré todo fácilmente", "la experiencia fue excelente",
        "compraría otra vez aquí",
    ]
    negativas_extra = [
        "no recomiendo este supermercado", "no los recomiendo",
        "no recomiendo comprar aquí", "no volvería a comprar aquí",
        "no volvería a este supermercado", "muy mala experiencia",
        "la atención fue pésima", "muy mala atención",
        "mala atención del personal", "no me gustó la atención",
        "no estoy satisfecho con la compra", "faltan productos básicos",
        "faltan muchos productos", "faltan productos de primera necesidad",
        "no encontré lo que buscaba", "no encontré los productos que buscaba",
        "los productos están vencidos", "los precios son demasiado altos",
        "demasiada cola para pagar", "la experiencia fue muy mala",
        "no los recomiendo porque faltan productos",
        "no recomiendo esta tienda porque faltan productos",
        "no volvería porque la atención fue mala",
        "no estoy satisfecho porque la atención fue pésima",
        "no recomiendo el supermercado porque hay pocos productos",
        "no encontré productos de primera necesidad",
        "faltan muchos productos básicos", "la atención fue terrible",
        "la atención fue muy mala",
        "los precios son demasiado altos y no recomiendo la tienda",
        "no volvería a comprar porque faltan productos",
        "no me gustó nada la atención", "la experiencia fue terrible",
        "no recomiendo este lugar", "no compraría nuevamente aquí",
    ]

    # Limpiar todas las frases
    datos = (
        [(f, "POSITIVO") for f in positivos_base]
        + [(f, "NEGATIVO") for f in negativos_base]
        + [(f, "POSITIVO") for f in positivas_extra]
        + [(f, "NEGATIVO") for f in negativas_extra]
    )

    textos = [_limpiar_texto_nlp(t) for t, _ in datos]
    etiquetas = [e for _, e in datos]

    # Intentar obtener también las opiniones del dataset
    try:
        df = cargar_df()
        invalidas = {"desconocido", "sin comentario", "???", "0"}
        opiniones_df = df["opinion_usuario"].dropna().astype(str).str.lower()
        opiniones_df = opiniones_df[~opiniones_df.isin(invalidas)]

        positivos_lower = {p.lower() for p in positivos_base}
        negativos_lower = {n.lower() for n in negativos_base}

        for op in opiniones_df:
            if op in positivos_lower:
                textos.append(_limpiar_texto_nlp(op))
                etiquetas.append("POSITIVO")
            elif op in negativos_lower:
                textos.append(_limpiar_texto_nlp(op))
                etiquetas.append("NEGATIVO")
    except Exception:
        pass

    # Entrenar TF-IDF + Regresión Logística
    X_train, X_test, y_train, y_test = train_test_split(
        textos, etiquetas, test_size=0.20, random_state=42, stratify=etiquetas
    )

    vectorizador = TfidfVectorizer(ngram_range=(1, 2))
    X_tr = vectorizador.fit_transform(X_train)
    X_te = vectorizador.transform(X_test)

    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_tr, y_train)

    precision = accuracy_score(y_test, modelo.predict(X_te))

    return vectorizador, modelo, round(precision * 100, 1)


def predecir_opinion(texto: str, vectorizador, modelo) -> dict:
    """Predice el sentimiento de una opinión y devuelve resultado + probabilidad."""
    limpio = _limpiar_texto_nlp(texto)
    vec = vectorizador.transform([limpio])
    resultado = modelo.predict(vec)[0]
    proba = modelo.predict_proba(vec)[0]
    idx = list(modelo.classes_).index(resultado)
    return {
        "sentimiento": resultado,
        "probabilidad": round(proba[idx] * 100, 1),
        "positivo_pct": round(proba[list(modelo.classes_).index("POSITIVO")] * 100, 1)
        if "POSITIVO" in modelo.classes_ else 50.0,
    }

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS DE INDICADORES
# ──────────────────────────────────────────────────────────────────────────────

def calcular_indicadores(df: pd.DataFrame) -> dict:
    return {
        "ventas_totales": df["total"].sum(),
        "venta_promedio": df["total"].mean(),
        "cantidad_total": df["cantidad"].sum(),
        "satisfaccion_prom": df["satisfaccion"].mean(),
        "num_transacciones": len(df),
        "producto_top": df.groupby("producto")["cantidad"].sum().idxmax(),
        "categoria_top": df.groupby("categoria")["total"].sum().idxmax(),
    }

# ──────────────────────────────────────────────────────────────────────────────
# GENERADOR DE RECOMENDACIONES
# ──────────────────────────────────────────────────────────────────────────────

def generar_recomendaciones(df: pd.DataFrame, nlp_vec, nlp_mod) -> list[str]:
    """
    Genera recomendaciones automáticas basadas en:
      - Tendencias de ventas semanales por producto
      - Predicción ML del mejor modelo
      - Análisis de sentimiento de las opiniones del dataset
      - Palabras clave en opiniones negativas
    """
    recos = []

    # ── 1. Producto con mayor crecimiento en fines de semana ──────────────────
    try:
        df_wk = df.copy()
        df_wk["dia_semana"] = df_wk["fecha"].dt.dayofweek
        fines = df_wk[df_wk["dia_semana"].isin([5, 6])]
        semana = df_wk[~df_wk["dia_semana"].isin([5, 6])]
        ventas_fin = fines.groupby("producto")["cantidad"].sum()
        ventas_sem = semana.groupby("producto")["cantidad"].sum()
        ratio = (ventas_fin / ventas_sem.replace(0, np.nan)).dropna()
        if not ratio.empty:
            prod_fin = ratio.idxmax()
            recos.append(
                f"El producto **{prod_fin}** presenta mayor crecimiento de ventas "
                f"durante los fines de semana (ratio finde/semana: {ratio.max():.2f}x)."
            )
    except Exception:
        pass

    # ── 2. Predicción ML ──────────────────────────────────────────────────────
    try:
        tabla_ml, df_ref = entrenar_ml()
        mejor_r2 = tabla_ml.loc[tabla_ml["R²"].idxmax()]
        prod_top = df_ref.groupby("producto")["cantidad"].sum().idxmax()
        mes_actual = pd.Timestamp.now().month
        año_actual = pd.Timestamp.now().year
        precio_hist = df_ref[df_ref["producto"].str.lower().str.title() == prod_top]["precio_unitario"].median()
        pred = ml_ventas.predecir_ventas(
            producto=prod_top,
            mes=mes_actual,
            año=año_actual,
            precio_unitario=float(precio_hist) if not np.isnan(precio_hist) else 10.0,
            descuento_pct=0.0,
            dia_semana=5,
            cliente_frecuente="si",
            satisfaccion=4.0,
            df_referencia=df_ref,
        )
        hist_prom = df_ref[df_ref["producto"].str.lower().str.title() == prod_top]["cantidad"].mean()
        if hist_prom > 0:
            incremento = round((pred["prediccion"] / hist_prom - 1) * 100, 1)
            signo = "incremento" if incremento >= 0 else "reducción"
            recos.append(
                f"El modelo **{mejor_r2['Modelo']}** (R²={mejor_r2['R²']}) predice un "
                f"{signo} del **{abs(incremento):.1f}%** en ventas de **{prod_top}** "
                f"para el próximo sábado respecto al promedio histórico."
            )
    except Exception:
        pass

    # ── 3. % Opiniones positivas del dataset ─────────────────────────────────
    try:
        opiniones_validas = df["opinion_usuario"].dropna().astype(str).str.lower()
        invalidas = {"desconocido", "sin comentario", "???", "0"}
        opiniones_validas = opiniones_validas[~opiniones_validas.isin(invalidas)]

        if len(opiniones_validas) > 0:
            resultados = [
                predecir_opinion(op, nlp_vec, nlp_mod)["sentimiento"]
                for op in opiniones_validas
            ]
            pct_pos = round(sum(r == "POSITIVO" for r in resultados) / len(resultados) * 100)
            recos.append(
                f"El **{pct_pos}%** de las opiniones registradas en el sistema "
                f"son **positivas** según el análisis de sentimiento."
            )
    except Exception:
        pass

    # ── 4. Detección de palabras negativas frecuentes ─────────────────────────
    try:
        palabras_negativas = [
            "retraso", "demora", "tarde", "vencido", "cola",
            "espera", "falta", "malo", "pésimo", "terrible"
        ]
        opiniones_str = " ".join(
            df["opinion_usuario"].dropna().astype(str).str.lower().tolist()
        )
        encontradas = [p for p in palabras_negativas if p in opiniones_str]
        if encontradas:
            recos.append(
                f"Se detectaron comentarios negativos con términos relacionados a: "
                f"**{', '.join(encontradas)}**. Se recomienda revisar estos aspectos operativos."
            )
    except Exception:
        pass

    # ── 5. Producto con mayor descuento vs. ventas ────────────────────────────
    try:
        desc_ventas = df.groupby("producto").agg(
            descuento_prom=("descuento_pct", "mean"),
            ventas_tot=("cantidad", "sum"),
        )
        prod_desc = desc_ventas["descuento_prom"].idxmax()
        pct_desc = desc_ventas.loc[prod_desc, "descuento_prom"]
        recos.append(
            f"**{prod_desc}** tiene el mayor descuento promedio ({pct_desc:.1f}%). "
            f"Evaluar si el margen justifica la estrategia de precio."
        )
    except Exception:
        pass

    # ── 6. Satisfacción baja por categoría ───────────────────────────────────
    try:
        sat_cat = df.groupby("categoria")["satisfaccion"].mean().sort_values()
        if not sat_cat.empty:
            cat_baja = sat_cat.index[0]
            val_baja = sat_cat.iloc[0]
            recos.append(
                f"La categoría **{cat_baja}** tiene la satisfacción promedio más baja "
                f"({val_baja:.2f}/5). Considerar acciones de mejora en calidad o atención."
            )
    except Exception:
        pass

    return recos

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — NAVEGACIÓN
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛒 Sistema SuperMercado")
    st.markdown("### Inteligencia Artificial")
    st.markdown("---")

    pagina = st.radio(
        "Navegación",
        options=[
            "📊 Dashboard",
            "🤖 Predicción de Ventas",
            "💬 Análisis de Opiniones",
            "🧠 Asistente de Decisiones",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("#### 📁 Dataset")
    df_global = cargar_df()
    st.markdown(f"- Registros: **{len(df_global):,}**")
    st.markdown(f"- Período: **{df_global['año_mes'].min()} → {df_global['año_mes'].max()}**")
    st.markdown(f"- Productos: **{df_global['producto'].nunique()}**")
    st.markdown("---")
    st.caption("SmartBusiness IA — SuperMercado")

# ──────────────────────────────────────────────────────────────────────────────
# PÁGINA 1 — DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────

if pagina == "📊 Dashboard":

    st.title("📊 Dashboard Estadístico")
    st.markdown("Indicadores clave y gráficos de rendimiento del supermercado.")

    df = cargar_df()
    ind = calcular_indicadores(df)

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Ventas Totales", f"S/ {ind['ventas_totales']:,.2f}")
    c2.metric("🧾 Ticket Promedio",  f"S/ {ind['venta_promedio']:,.2f}")
    c3.metric("📦 Unidades Vendidas", f"{ind['cantidad_total']:,.0f}")
    c4.metric("⭐ Satisfacción",      f"{ind['satisfaccion_prom']:.2f} / 5")
    c5.metric("🔄 Transacciones",     f"{ind['num_transacciones']:,}")

    st.markdown("---")

    c6, c7 = st.columns(2)
    with c6:
        st.info(f"🏆 **Producto más vendido:** {ind['producto_top']}")
    with c7:
        st.info(f"📂 **Categoría con mayores ingresos:** {ind['categoria_top'].title()}")

    st.markdown("---")
    st.subheader("Gráficos Estadísticos")

    # Fila 1
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        grafico_ventas_por_mes(df, ax1)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        grafico_productos_mas_vendidos(df, ax2)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # Fila 2
    col3, col4 = st.columns(2)

    with col3:
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        grafico_ventas_por_categoria(df, ax3)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    with col4:
        fig4, ax4 = plt.subplots(figsize=(7, 4))
        grafico_distribucion_precios(df, ax4)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    # Fila 3
    col5, col6 = st.columns(2)

    with col5:
        fig5, ax5 = plt.subplots(figsize=(7, 4))
        grafico_correlacion(df, ax5)
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

    with col6:
        fig6, ax6 = plt.subplots(figsize=(7, 4))
        grafico_histograma_satisfaccion(df, ax6)
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close(fig6)

    # Tabla resumen
    st.markdown("---")
    st.subheader("Ventas por Producto")
    tabla_prod = (
        df.groupby("producto")
        .agg(
            Unidades=("cantidad", "sum"),
            Ingresos=("total", "sum"),
            Precio_Prom=("precio_unitario", "mean"),
            Satisfaccion_Prom=("satisfaccion", "mean"),
        )
        .sort_values("Ingresos", ascending=False)
        .reset_index()
    )
    tabla_prod["Ingresos"] = tabla_prod["Ingresos"].map("S/ {:,.2f}".format)
    tabla_prod["Precio_Prom"] = tabla_prod["Precio_Prom"].map("S/ {:,.2f}".format)
    tabla_prod["Satisfaccion_Prom"] = tabla_prod["Satisfaccion_Prom"].map("{:.2f}".format)
    st.dataframe(tabla_prod, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# PÁGINA 2 — PREDICCIÓN DE VENTAS
# ──────────────────────────────────────────────────────────────────────────────

elif pagina == "🤖 Predicción de Ventas":

    st.title("🤖 Predicción de Ventas")
    st.markdown(
        "Ingresá los parámetros del producto y el sistema predice "
        "las unidades que se venderán usando el mejor modelo de ML."
    )

    # Entrenamiento (cacheado)
    with st.spinner("Preparando modelos…"):
        tabla_modelos, df_ref = entrenar_ml()

    # Métricas de modelos
    st.subheader("Comparativa de Modelos")
    col_m1, col_m2, col_m3 = st.columns(3)
    for i, (_, row) in enumerate(tabla_modelos.iterrows()):
        col = [col_m1, col_m2, col_m3][i]
        mejor = "✅ " if row["Modelo"] == tabla_modelos.loc[tabla_modelos["R²"].idxmax(), "Modelo"] else ""
        col.metric(
            f"{mejor}{row['Modelo']}",
            f"R² = {row['R²']}",
            f"MAE: {row['MAE']} | RMSE: {row['RMSE']}",
        )

    st.markdown("---")
    st.subheader("Parámetros de Predicción")

    productos_disponibles = sorted(df_ref["producto"].unique().tolist())
    meses_nombre = [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
    ]
    dias_nombre = [
        "Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"
    ]

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        producto = st.selectbox("Producto", productos_disponibles)
        mes = st.selectbox(
            "Mes",
            options=list(range(1, 13)),
            format_func=lambda x: meses_nombre[x - 1],
            index=pd.Timestamp.now().month - 1,
        )
        año = st.number_input(
            "Año",
            min_value=2020,
            max_value=2030,
            value=pd.Timestamp.now().year,
        )

    with col_b:
        precio_hist = df_ref[df_ref["producto"] == producto]["precio_unitario"].median()
        precio_unitario = st.number_input(
            "Precio Unitario (S/)",
            min_value=0.1,
            max_value=500.0,
            value=float(precio_hist) if not np.isnan(precio_hist) else 10.0,
            step=0.5,
        )
        descuento_pct = st.slider("Descuento (%)", 0, 100, 0)
        satisfaccion = st.slider("Satisfacción esperada (1–5)", 1.0, 5.0, 4.0, step=0.5)

    with col_c:
        dia_semana = st.selectbox(
            "Día de la semana",
            options=list(range(7)),
            format_func=lambda x: dias_nombre[x],
            index=5,
        )
        cliente_frecuente = st.radio(
            "¿Cliente frecuente?",
            options=["si", "no"],
            horizontal=True,
        )

    st.markdown("")
    if st.button("🔍 Predecir Ventas", type="primary", use_container_width=True):
        try:
            resultado = ml_ventas.predecir_ventas(
                producto=producto,
                mes=mes,
                año=año,
                precio_unitario=precio_unitario,
                descuento_pct=float(descuento_pct),
                dia_semana=dia_semana,
                cliente_frecuente=cliente_frecuente,
                satisfaccion=satisfaccion,
                df_referencia=df_ref,
            )

            st.markdown("---")
            st.subheader("Resultado de la Predicción")

            r1, r2, r3 = st.columns(3)
            r1.metric(
                "📦 Unidades predichas",
                f"{resultado['prediccion']:.0f} uds",
            )
            r2.metric(
                "🤖 Modelo usado",
                resultado["modelo_usado"],
            )
            hist = resultado["detalles"]["hist_ventas_prod"]
            if hist > 0:
                variacion = round((resultado["prediccion"] / hist - 1) * 100, 1)
                r3.metric(
                    "📈 Variación vs. histórico",
                    f"{variacion:+.1f}%",
                    delta=f"{variacion:+.1f}%",
                )
            else:
                r3.metric("📊 Histórico prod.", "Sin datos previos")

            # Detalles expandibles
            with st.expander("Ver detalles completos"):
                det = resultado["detalles"]
                st.markdown(f"""
| Parámetro | Valor |
|---|---|
| Producto | {det['producto']} |
| Mes | {meses_nombre[det['mes']-1]} {det['año']} |
| Día | {dias_nombre[det['dia_semana']]} |
| Precio Unitario | S/ {det['precio_unitario']} |
| Descuento | {det['descuento_pct']}% |
| Cliente Frecuente | {det['cliente_frecuente']} |
| Satisfacción | {det['satisfaccion']}/5 |
| Ventas hist. producto | {det['hist_ventas_prod']} uds (prom.) |
| Ventas hist. mes | {det['hist_ventas_mes']} uds (prom.) |
                """)

        except Exception as e:
            st.error(f"Error en la predicción: {e}")

    # Gráfico de histórico del producto seleccionado
    st.markdown("---")
    st.subheader(f"Histórico de Ventas — {producto}")
    hist_prod = (
        df_ref[df_ref["producto"] == producto]
        .groupby("año_mes")["cantidad"]
        .sum()
        .reset_index()
        .sort_values("año_mes")
    )
    if not hist_prod.empty:
        fig_h, ax_h = plt.subplots(figsize=(10, 3.5))
        ax_h.plot(
            hist_prod["año_mes"],
            hist_prod["cantidad"],
            marker="o",
            linewidth=2,
            color="#2980b9",
        )
        ax_h.fill_between(
            range(len(hist_prod)),
            hist_prod["cantidad"],
            alpha=0.15,
            color="#2980b9",
        )
        ticks = range(0, len(hist_prod), max(1, len(hist_prod) // 8))
        ax_h.set_xticks(list(ticks))
        ax_h.set_xticklabels(
            [hist_prod["año_mes"].iloc[i] for i in ticks],
            rotation=45, ha="right", fontsize=8
        )
        ax_h.set_title(f"Ventas mensuales — {producto}")
        ax_h.set_ylabel("Unidades")
        ax_h.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_h)
        plt.close(fig_h)
    else:
        st.info("Sin datos históricos para este producto.")

# ──────────────────────────────────────────────────────────────────────────────
# PÁGINA 3 — ANÁLISIS DE OPINIONES
# ──────────────────────────────────────────────────────────────────────────────

elif pagina == "💬 Análisis de Opiniones":

    st.title("💬 Análisis de Opiniones")
    st.markdown(
        "Clasifica opiniones de clientes como **POSITIVO** o **NEGATIVO** "
        "usando un modelo TF-IDF + Regresión Logística (mismo algoritmo que `nlp_opiniones.py`)."
    )

    with st.spinner("Cargando modelo NLP…"):
        nlp_vec, nlp_mod, nlp_precision = entrenar_nlp()

    st.success(f"Modelo listo — Precisión en prueba: **{nlp_precision}%**")

    # ── Análisis de una opinión nueva ────────────────────────────────────────
    st.subheader("Clasificar una nueva opinión")

    opinion_input = st.text_area(
        "Escribe una opinión del cliente:",
        placeholder="Ej: Muy buena atención, encontré todo lo que necesitaba…",
        height=100,
    )

    if st.button("🔍 Analizar Opinión", type="primary"):
        if opinion_input.strip():
            res = predecir_opinion(opinion_input, nlp_vec, nlp_mod)
            color = "🟢" if res["sentimiento"] == "POSITIVO" else "🔴"
            st.markdown(f"""
<div style="background:#f8f9fa; border-radius:10px; padding:18px; border-left:5px solid {'#27ae60' if res['sentimiento']=='POSITIVO' else '#e74c3c'}; margin-top:12px;">
  <h4 style="margin:0 0 8px 0;">{color} Resultado: {res['sentimiento']}</h4>
  <p style="margin:0; font-size:15px;">Probabilidad: <strong>{res['probabilidad']}%</strong></p>
</div>
""", unsafe_allow_html=True)
        else:
            st.warning("Por favor ingresá una opinión.")

    # ── Análisis masivo del dataset ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("Análisis masivo del dataset")

    df = cargar_df()
    invalidas = {"desconocido", "sin comentario", "???", "0"}
    opiniones_df = (
        df["opinion_usuario"]
        .dropna()
        .astype(str)
        .str.lower()
    )
    opiniones_df = opiniones_df[~opiniones_df.isin(invalidas)]

    if len(opiniones_df) > 0:
        with st.spinner("Analizando opiniones del dataset…"):
            resultados_masivos = [
                predecir_opinion(op, nlp_vec, nlp_mod)
                for op in opiniones_df
            ]

        total_op = len(resultados_masivos)
        positivos_n = sum(r["sentimiento"] == "POSITIVO" for r in resultados_masivos)
        negativos_n = total_op - positivos_n
        pct_pos = round(positivos_n / total_op * 100, 1)
        pct_neg = round(negativos_n / total_op * 100, 1)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total opiniones analizadas", f"{total_op}")
        m2.metric("🟢 Positivas", f"{positivos_n} ({pct_pos}%)")
        m3.metric("🔴 Negativas", f"{negativos_n} ({pct_neg}%)")

        # Gráfico de torta
        fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
        ax_pie.pie(
            [positivos_n, negativos_n],
            labels=["Positivas", "Negativas"],
            colors=["#27ae60", "#e74c3c"],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        ax_pie.set_title("Distribución de Sentimientos")
        col_pie, col_bar = st.columns(2)
        with col_pie:
            plt.tight_layout()
            st.pyplot(fig_pie)
            plt.close(fig_pie)

        # Gráfico de barras por producto
        with col_bar:
            df_temp = df.copy()
            df_temp = df_temp[~df_temp["opinion_usuario"].fillna("").str.lower().isin(invalidas)]
            df_temp = df_temp.dropna(subset=["opinion_usuario"])
            df_temp["sentimiento_pred"] = [r["sentimiento"] for r in resultados_masivos]

            sent_prod = (
                df_temp.groupby(["producto", "sentimiento_pred"])
                .size()
                .unstack(fill_value=0)
                .reset_index()
            )
            fig_sp, ax_sp = plt.subplots(figsize=(6, 4))
            sent_prod_plot = sent_prod.set_index("producto")
            colores_sent = {
                "POSITIVO": "#27ae60",
                "NEGATIVO": "#e74c3c",
            }
            for col_sent in sent_prod_plot.columns:
                ax_sp.bar(
                    sent_prod_plot.index,
                    sent_prod_plot[col_sent],
                    label=col_sent,
                    color=colores_sent.get(col_sent, "#888"),
                    alpha=0.8,
                )
            ax_sp.set_title("Sentimientos por Producto")
            ax_sp.set_xlabel("Producto")
            ax_sp.set_ylabel("Cantidad")
            ax_sp.legend()
            ax_sp.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            st.pyplot(fig_sp)
            plt.close(fig_sp)

        # Tabla de opiniones clasificadas
        st.markdown("---")
        st.subheader("Detalle de Opiniones Clasificadas")
        df_tabla_op = pd.DataFrame({
            "Opinión": opiniones_df.values,
            "Sentimiento": [r["sentimiento"] for r in resultados_masivos],
            "Confianza (%)": [r["probabilidad"] for r in resultados_masivos],
        })
        st.dataframe(df_tabla_op, use_container_width=True, hide_index=True)

    else:
        st.info("No hay opiniones válidas en el dataset para analizar.")

    # ── Prueba por lotes ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Probar múltiples opiniones")
    st.markdown("Ingresá una opinión por línea:")
    batch_text = st.text_area("Opiniones (una por línea):", height=120)
    if st.button("Analizar lote"):
        lineas = [l.strip() for l in batch_text.split("\n") if l.strip()]
        if lineas:
            resultados_lote = [
                {
                    "Opinión": op,
                    **predecir_opinion(op, nlp_vec, nlp_mod),
                }
                for op in lineas
            ]
            df_lote = pd.DataFrame(resultados_lote)[
                ["Opinión", "sentimiento", "probabilidad"]
            ]
            df_lote.columns = ["Opinión", "Sentimiento", "Confianza (%)"]
            st.dataframe(df_lote, use_container_width=True, hide_index=True)
        else:
            st.warning("Ingresá al menos una opinión.")

# ──────────────────────────────────────────────────────────────────────────────
# PÁGINA 4 — ASISTENTE DE DECISIONES
# ──────────────────────────────────────────────────────────────────────────────

elif pagina == "🧠 Asistente de Decisiones":

    st.title("🧠 Asistente de Decisiones")
    st.markdown(
        "Recomendaciones automáticas generadas a partir del análisis de datos, "
        "predicciones de ventas y sentimiento de opiniones."
    )

    df = cargar_df()

    with st.spinner("Generando recomendaciones…"):
        nlp_vec, nlp_mod, _ = entrenar_nlp()
        recos = generar_recomendaciones(df, nlp_vec, nlp_mod)

    # ── Cuadro de recomendaciones ─────────────────────────────────────────────
    st.markdown("""
<div class="reco-box">
  <div class="reco-title">📋 RECOMENDACIONES</div>
""" + "".join(
        f'<div style="margin-bottom:8px;">➤ {r}</div>'
        for r in recos
    ) + """
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Panel de análisis rápido ──────────────────────────────────────────────
    st.subheader("Análisis Rápido por Dimensión")

    tab1, tab2, tab3 = st.tabs(["📦 Ventas", "⭐ Satisfacción", "📅 Temporalidad"])

    with tab1:
        prod_ventas = (
            df.groupby("producto")
            .agg(
                Unidades=("cantidad", "sum"),
                Ingresos=("total", "sum"),
            )
            .sort_values("Ingresos", ascending=False)
            .reset_index()
        )
        prod_ventas["Ingresos"] = prod_ventas["Ingresos"].map("S/ {:,.2f}".format)
        st.dataframe(prod_ventas, use_container_width=True, hide_index=True)

    with tab2:
        sat_prod = (
            df.groupby("producto")["satisfaccion"]
            .agg(["mean", "min", "max", "count"])
            .rename(columns={"mean": "Promedio", "min": "Mínimo", "max": "Máximo", "count": "Registros"})
            .sort_values("Promedio")
            .reset_index()
        )
        sat_prod["Promedio"] = sat_prod["Promedio"].map("{:.2f}".format)
        st.dataframe(sat_prod, use_container_width=True, hide_index=True)

        fig_sat, ax_sat = plt.subplots(figsize=(8, 3.5))
        sat_data = df.groupby("producto")["satisfaccion"].mean().sort_values()
        colores_sat = ["#e74c3c" if v < 3 else "#f39c12" if v < 4 else "#27ae60" for v in sat_data.values]
        ax_sat.barh(sat_data.index, sat_data.values, color=colores_sat)
        ax_sat.axvline(3, color="gray", linestyle="--", linewidth=1, label="Umbral 3.0")
        ax_sat.set_title("Satisfacción promedio por producto")
        ax_sat.set_xlabel("Puntuación")
        ax_sat.legend()
        plt.tight_layout()
        st.pyplot(fig_sat)
        plt.close(fig_sat)

    with tab3:
        df_temp = df.copy()
        df_temp["dia_semana"] = df_temp["fecha"].dt.dayofweek
        df_temp["nombre_dia"] = df_temp["fecha"].dt.day_name()
        ventas_dia = (
            df_temp.groupby(["dia_semana", "nombre_dia"])["cantidad"]
            .sum()
            .reset_index()
            .sort_values("dia_semana")
        )

        fig_dia, ax_dia = plt.subplots(figsize=(8, 3.5))
        colores_dia = [
            "#e74c3c" if d in [5, 6] else "#3498db"
            for d in ventas_dia["dia_semana"]
        ]
        ax_dia.bar(ventas_dia["nombre_dia"], ventas_dia["cantidad"], color=colores_dia)
        ax_dia.set_title("Ventas por día de la semana")
        ax_dia.set_xlabel("Día")
        ax_dia.set_ylabel("Unidades vendidas")
        ax_dia.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        st.pyplot(fig_dia)
        plt.close(fig_dia)

        st.markdown(
            "_Rojo = fin de semana | Azul = día de semana_",
            unsafe_allow_html=False,
        )

    # ── Consulta personalizada ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔎 Consulta de Producto")
    st.markdown("Seleccioná un producto para ver un resumen ejecutivo.")

    producto_sel = st.selectbox(
        "Producto",
        options=sorted(df["producto"].unique()),
        key="asistente_producto",
    )

    df_prod = df[df["producto"] == producto_sel]
    if not df_prod.empty:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Unidades vendidas", f"{df_prod['cantidad'].sum():,.0f}")
        p2.metric("Ingresos totales", f"S/ {df_prod['total'].sum():,.2f}")
        p3.metric("Precio promedio", f"S/ {df_prod['precio_unitario'].mean():,.2f}")
        p4.metric("Satisfacción", f"{df_prod['satisfaccion'].mean():.2f}/5")

        # Tendencia mensual del producto
        tend_prod = (
            df_prod.groupby("año_mes")["cantidad"]
            .sum()
            .reset_index()
            .sort_values("año_mes")
        )
        fig_tp, ax_tp = plt.subplots(figsize=(9, 3))
        ax_tp.plot(
            tend_prod["año_mes"], tend_prod["cantidad"],
            marker="o", linewidth=2, color="#8e44ad"
        )
        ax_tp.fill_between(
            range(len(tend_prod)), tend_prod["cantidad"],
            alpha=0.12, color="#8e44ad"
        )
        ticks_tp = range(0, len(tend_prod), max(1, len(tend_prod) // 8))
        ax_tp.set_xticks(list(ticks_tp))
        ax_tp.set_xticklabels(
            [tend_prod["año_mes"].iloc[i] for i in ticks_tp],
            rotation=45, ha="right", fontsize=8
        )
        ax_tp.set_title(f"Tendencia mensual — {producto_sel}")
        ax_tp.set_ylabel("Unidades")
        ax_tp.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_tp)
        plt.close(fig_tp)
