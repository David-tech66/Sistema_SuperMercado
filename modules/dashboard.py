import os                           # Importarmos os para manejar rutas de archivos 
import warnings                     # Importamos warnings para suprimir advertencias innecesarias
import matplotlib.pyplot as plt     # Importamos matplotlib.pyplot para crear gráficos
import matplotlib.ticker as mticker # Importamos matplotlib.ticker para formatear los ejes de los gráficos
import seaborn as sns               # Importamos seaborn para crear gráficos estadísticos más atractivos
import pandas as pd                 # Importamos pandas para manejar y analizar datos en forma de DataFrame
import numpy as np                  # Importamos numpy para operaciones numéricas y manejo de arrays

warnings.filterwarnings("ignore")   # Suprimimos advertencias de matplotlib y seaborn para mantener la salida limpia

# CONFIGURACIÓN VISUAL GLOBAL
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.facecolor": "#f9f9f9",
    "axes.facecolor":   "#f9f9f9",
    "axes.titleweight": "bold",
    "axes.titlesize":   13,
})

PALETA_CATEGORIAS = "Set2"
PALETA_PRODUCTOS  = "Blues_d"
COLOR_HIST        = "#4C72B0"
COLOR_LINE        = "#DD8452"


# 1. CARGAR DATOS
def cargar_datos() -> pd.DataFrame:
    """
    Carga supermercado_limpio.csv usando rutas relativas
    al módulo, igual que en ml_ventas.py, para que funcione
    sin importar desde qué directorio se ejecute el script.
    """
    candidatas = [
        "supermercado_limpio.csv",
        "../supermercado_limpio.csv",
        os.path.join(os.path.dirname(__file__), "..", "supermercado_limpio.csv"),
    ]
    ruta = None
    for c in candidatas:
        if os.path.exists(c):
            ruta = c
            break

    if ruta is None:
        raise FileNotFoundError(
            "No se encontró 'supermercado_limpio.csv'.\n"
            "Ejecuta primero supermercado2.py para generarlo."
        )

    df = pd.read_csv(ruta, encoding="utf-8-sig")
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    return df


# ==========================================
# 2. GRÁFICAS
# ==========================================

# ------------------------------------------
# 2.1 Ventas por mes (línea de tendencia)
# ------------------------------------------

def grafico_ventas_por_mes(df: pd.DataFrame, ax: plt.Axes) -> None:
    """
    Suma del total de ventas agrupado por año_mes.
    Muestra la tendencia completa del período.
    """
    ventas_mes = (
        df.groupby("año_mes")["total"]
        .sum()
        .reset_index()
        .sort_values("año_mes")
    )

    ax.plot(
        ventas_mes["año_mes"],
        ventas_mes["total"],
        color=COLOR_LINE,
        marker="o",
        linewidth=2,
        markersize=5,
    )
    ax.fill_between(
        range(len(ventas_mes)),
        ventas_mes["total"],
        alpha=0.15,
        color=COLOR_LINE,
    )

    # Etiquetas de eje X: mostrar solo cada 3 meses para no saturar
    ticks = range(0, len(ventas_mes), 3)
    ax.set_xticks(list(ticks))
    ax.set_xticklabels(
        [ventas_mes["año_mes"].iloc[i] for i in ticks],
        rotation=45,
        ha="right",
        fontsize=9,
    )

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"S/ {x:,.0f}")
    )
    ax.set_title("Ventas por Mes")
    ax.set_xlabel("Período")
    ax.set_ylabel("Total de Ventas (S/)")


# ------------------------------------------
# 2.2 Productos más vendidos (barras)
# ------------------------------------------

def grafico_productos_mas_vendidos(df: pd.DataFrame, ax: plt.Axes) -> None:
    """
    Top 10 productos por unidades vendidas (suma de cantidad).
    Excluye la categoría 'desconocido'.
    """
    datos = (
        df[df["producto"] != "Desconocido"]
        .groupby("producto")["cantidad"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    sns.barplot(
        data=datos,
        x="cantidad",
        y="producto",
        palette=PALETA_PRODUCTOS,
        ax=ax,
    )

    # Etiquetas de valor al final de cada barra
    for i, (_, fila) in enumerate(datos.iterrows()):
        ax.text(
            fila["cantidad"] + datos["cantidad"].max() * 0.01,
            i,
            f"{fila['cantidad']:,.0f}",
            va="center",
            fontsize=9,
        )

    ax.set_title("Productos más Vendidos")
    ax.set_xlabel("Unidades Vendidas")
    ax.set_ylabel("Producto")
    ax.set_xlim(0, datos["cantidad"].max() * 1.12)


# ------------------------------------------
# 2.3 Ventas por categoría (barras)
# ------------------------------------------

def grafico_ventas_por_categoria(df: pd.DataFrame, ax: plt.Axes) -> None:
    """
    Total de ingresos (suma de 'total') por categoría,
    excluyendo 'desconocido'.
    """
    datos = (
        df[df["categoria"] != "desconocido"]
        .groupby("categoria")["total"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    colores = sns.color_palette(PALETA_CATEGORIAS, len(datos))
    bars = ax.bar(
        datos["categoria"],
        datos["total"],
        color=colores,
        edgecolor="white",
        linewidth=0.8,
    )

    # Etiquetas sobre cada barra
    for bar, valor in zip(bars, datos["total"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + datos["total"].max() * 0.01,
            f"S/ {valor:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"S/ {x:,.0f}")
    )
    ax.set_title("Ventas por Categoría")
    ax.set_xlabel("Categoría")
    ax.set_ylabel("Ingresos Totales (S/)")
    ax.tick_params(axis="x", rotation=20)


# ------------------------------------------
# 2.4 Distribución de precios (KDE + hist)
# ------------------------------------------

def grafico_distribucion_precios(df: pd.DataFrame, ax: plt.Axes) -> None:
    """
    Distribución de 'precio_unitario' mediante histograma
    con curva de densidad (KDE).
    """
    precios = df["precio_unitario"].dropna()

    sns.histplot(
        precios,
        bins=20,
        color=COLOR_HIST,
        kde=True,
        alpha=0.7,
        line_kws={"linewidth": 2},
        ax=ax,
    )

    # Líneas de referencia estadísticas
    media   = precios.mean()
    mediana = precios.median()
    ax.axvline(media,   color="#d62728", linestyle="--", linewidth=1.5, label=f"Media   S/ {media:.2f}")
    ax.axvline(mediana, color="#2ca02c", linestyle=":",  linewidth=1.5, label=f"Mediana S/ {mediana:.2f}")
    ax.legend(fontsize=9)

    ax.set_title("Distribución de Precios Unitarios")
    ax.set_xlabel("Precio Unitario (S/)")
    ax.set_ylabel("Frecuencia")


# ------------------------------------------
# 2.5 Correlación entre variables (heatmap)
# ------------------------------------------

def grafico_correlacion(df: pd.DataFrame, ax: plt.Axes) -> None:
    """
    Mapa de calor de la matriz de correlación de Pearson
    entre las variables numéricas clave del dataset.
    """
    columnas_numericas = [
        "cantidad",
        "precio_unitario",
        "descuento_pct",
        "satisfaccion",
        "total",
    ]

    corr = df[columnas_numericas].dropna().corr()

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title("Correlación entre Variables Numéricas")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)


# ------------------------------------------
# 2.6 Histograma — Satisfacción del cliente
# ------------------------------------------

def grafico_histograma_satisfaccion(df: pd.DataFrame, ax: plt.Axes) -> None:
    """
    Distribución de las puntuaciones de satisfacción (1–5).
    Muestra el porcentaje sobre cada barra.
    """
    datos = df["satisfaccion"].dropna()

    # Calcular frecuencias para porcentajes
    conteos = datos.value_counts().sort_index()
    total   = conteos.sum()

    colores = sns.color_palette("RdYlGn", len(conteos))
    bars = ax.bar(
        conteos.index,
        conteos.values,
        color=colores,
        edgecolor="white",
        linewidth=0.8,
        width=0.6,
    )

    # Porcentaje sobre cada barra
    for bar, valor in zip(bars, conteos.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.003,
            f"{valor / total * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(conteos.index)
    ax.set_xticklabels(
        [f"{'★' * int(v)}" for v in conteos.index],
        fontsize=10,
    )
    ax.set_title("Histograma – Satisfacción del Cliente")
    ax.set_xlabel("Puntuación (1–5 estrellas)")
    ax.set_ylabel("Número de Transacciones")
    ax.set_xlim(0.3, 5.7)


# ==========================================
# 3. ARMAR Y MOSTRAR EL DASHBOARD
# ==========================================

def mostrar_dashboard() -> None:
    """
    Genera el dashboard completo con los 6 gráficos
    en un layout 3×2 y lo muestra en pantalla.
    """
    df = cargar_datos()

    print("=" * 52)
    print("         DASHBOARD ESTADÍSTICO")
    print("         Sistema Supermercado")
    print("=" * 52)
    print(f"  Registros cargados : {len(df):,}")
    print(f"  Período            : {df['año_mes'].min()} → {df['año_mes'].max()}")
    print(f"  Categorías         : {', '.join(sorted(df['categoria'].unique()))}")
    print(f"  Ventas totales     : S/ {df['total'].sum():,.2f}")
    print(f"  Satisfacción prom. : {df['satisfaccion'].mean():.2f} / 5")
    print("=" * 52)
    print("  Generando gráficos…")

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(16, 18),
        facecolor="#f9f9f9",
    )
    fig.suptitle(
        "Dashboard Estadístico – Sistema Supermercado",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )

    grafico_ventas_por_mes          (df, axes[0, 0])
    grafico_productos_mas_vendidos  (df, axes[0, 1])
    grafico_ventas_por_categoria    (df, axes[1, 0])
    grafico_distribucion_precios    (df, axes[1, 1])
    grafico_correlacion             (df, axes[2, 0])
    grafico_histograma_satisfaccion (df, axes[2, 1])

    plt.tight_layout()
    plt.show()

    print("  Dashboard mostrado correctamente.")


# ==========================================
# PUNTO DE ENTRADA
# ==========================================

if __name__ == "__main__":
    mostrar_dashboard()
