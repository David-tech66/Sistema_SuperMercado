import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# 1. VENTAS POR MES
def mostrar_ventas_por_mes(df):
    st.subheader("Tendencia de Ventas por Mes")

    df = df.copy()
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['mes_anio'] = df['fecha'].dt.to_period('M').astype(str)
    ventas_mes = df.groupby('mes_anio')['total'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(
        data=ventas_mes,
        x='mes_anio',
        y='total',
        marker='o',
        color='#1f77b4',
        ax=ax,
        linewidth=2.5
    )

    ax.set_title("Ingresos Totales por Mes", fontsize=14, fontweight='bold')
    ax.set_xlabel("Mes y Año")
    ax.set_ylabel("Ingresos (S/)")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# ==========================================
# 2. PRODUCTOS MÁS VENDIDOS
# ==========================================

def mostrar_productos_top(df):
    st.subheader("🏆 Top Productos Más Vendidos")

    top_productos = (
        df.groupby('producto')['cantidad']
        .sum()
        .reset_index()
        .sort_values(by='cantidad', ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=top_productos,
        x='cantidad',
        y='producto',
        palette='Greens_r',
        ax=ax
    )

    ax.set_title("Top 10 Productos con Mayor Demanda", fontsize=14, fontweight='bold')
    ax.set_xlabel("Unidades Vendidas")
    ax.set_ylabel("Producto")

    # Etiquetas de valor en cada barra
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', padding=4, fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ==========================================
# 3. VENTAS POR CATEGORÍA
# ==========================================

def mostrar_ventas_por_categoria(df):
    st.subheader("🗂️ Ventas por Categoría")

    ventas_cat = (
        df[df['categoria'] != 'desconocido']
        .groupby('categoria')['total']
        .sum()
        .reset_index()
        .sort_values(by='total', ascending=False)
    )

    col1, col2 = st.columns(2)

    # Gráfico de barras
    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(
            data=ventas_cat,
            x='total',
            y='categoria',
            palette='Blues_r',
            ax=ax
        )
        ax.set_title("Ingresos por Categoría", fontsize=13, fontweight='bold')
        ax.set_xlabel("Ingresos Totales (S/)")
        ax.set_ylabel("Categoría")

        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f', padding=4, fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Gráfico circular (pie)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.pie(
            ventas_cat['total'],
            labels=ventas_cat['categoria'],
            autopct='%1.1f%%',
            startangle=140,
            colors=sns.color_palette('Blues_r', len(ventas_cat))
        )
        ax.set_title("Participación por Categoría", fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ==========================================
# 4. DISTRIBUCIÓN DE PRECIOS
# ==========================================

def mostrar_distribucion_precios(df):
    st.subheader("💰 Distribución de Precios Unitarios")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histograma de precio_unitario
    sns.histplot(
        data=df,
        x='precio_unitario',
        bins=30,
        kde=True,
        color='#e07b39',
        ax=axes[0]
    )
    axes[0].set_title("Distribución del Precio Unitario", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Precio Unitario (S/)")
    axes[0].set_ylabel("Frecuencia")
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Boxplot por categoría
    df_filtrado = df[df['categoria'] != 'desconocido'].copy()
    sns.boxplot(
        data=df_filtrado,
        x='precio_unitario',
        y='categoria',
        palette='Oranges',
        ax=axes[1]
    )
    axes[1].set_title("Precio Unitario por Categoría", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Precio Unitario (S/)")
    axes[1].set_ylabel("Categoría")
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ==========================================
# 5. CORRELACIÓN ENTRE VARIABLES
# ==========================================

def mostrar_correlacion(df):
    st.subheader("🔗 Correlación entre Variables Numéricas")

    columnas_numericas = [
        'cantidad',
        'precio_unitario',
        'descuento_pct',
        'satisfaccion',
        'total'
    ]

    df_num = df[columnas_numericas].dropna()
    matriz_corr = df_num.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matriz_corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        linewidths=0.5,
        square=True,
        ax=ax
    )
    ax.set_title("Mapa de Correlación", fontsize=14, fontweight='bold')
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Valores cercanos a 1 indican correlación positiva, "
        "cercanos a -1 indican correlación negativa, "
        "y cercanos a 0 indican poca relación lineal."
    )


# ==========================================
# 6. HISTOGRAMA DE VENTAS TOTALES
# ==========================================

def mostrar_histograma_ventas(df):
    st.subheader("📊 Histograma de Ventas Totales")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histograma del total por transacción
    sns.histplot(
        data=df,
        x='total',
        bins=35,
        kde=True,
        color='#9467bd',
        ax=axes[0]
    )
    axes[0].set_title("Distribución del Total por Venta", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Total (S/)")
    axes[0].set_ylabel("Frecuencia")
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Histograma de satisfacción
    sns.histplot(
        data=df,
        x='satisfaccion',
        bins=5,
        kde=False,
        color='#d62728',
        discrete=True,
        ax=axes[1]
    )
    axes[1].set_title("Distribución de Satisfacción del Cliente", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Puntuación de Satisfacción (1–5)")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_xticks([1, 2, 3, 4, 5])
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ==========================================
# ZONA DE PRUEBA (Solo ejecución directa)
# ==========================================

if __name__ == "__main__":
    import os

    st.set_page_config(
        page_title="Dashboard Supermercado",
        page_icon="🛒",
        layout="wide"
    )

    st.title("🛒 Dashboard Estadístico — Vista de Prueba")
    st.write("Esta vista es solo para prueba local del módulo.")

    # Rutas candidatas al CSV limpio
    rutas_candidatas = [
        'supermercado_limpio.csv',
        '../supermercado_limpio.csv',
        'Sistema_SuperMercado/supermercado_limpio.csv',
    ]

    df_prueba = None
    for ruta in rutas_candidatas:
        if os.path.exists(ruta):
            df_prueba = pd.read_csv(ruta)
            break

    if df_prueba is not None:
        mostrar_ventas_por_mes(df_prueba)
        mostrar_productos_top(df_prueba)
        mostrar_ventas_por_categoria(df_prueba)
        mostrar_distribucion_precios(df_prueba)
        mostrar_correlacion(df_prueba)
        mostrar_histograma_ventas(df_prueba)
    else:
        st.error(
            "No se encontró el archivo 'supermercado_limpio.csv'. "
            "Asegúrate de ejecutar primero supermercado2.py para generarlo."
        )
