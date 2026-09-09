import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configurar el estilo visual global de Seaborn
sns.set_theme(style="whitegrid")

# 1. FUNCIONES DE GRÁFICOS
# ------------------------------------------
def grafico_ventas_por_mes(df: pd.DataFrame, ax: plt.Axes) -> None:
    ventas_mes = df.groupby("año_mes")["total"].sum().reset_index().sort_values("año_mes")
    
    # Usamos Grafico Lineal
    sns.lineplot(
        data=ventas_mes, 
        x="año_mes", 
        y="total", 
        marker="o", 
        linewidth=2, 
        ax=ax, 
        color="#1f77b4"
    )
    ax.fill_between(
        range(len(ventas_mes)), 
        ventas_mes["total"], 
        alpha=0.15, 
        color="#1f77b4"
    )
    # Etiquetas de eje X: mostrar solo cada 3 meses para no saturar
    ticks = range(0, len(ventas_mes), 3)
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([ventas_mes["año_mes"].iloc[i] for i in ticks], rotation=45, ha="right", fontsize=9)
    ax.set_title("Ventas por Mes", fontweight='bold')
    ax.set_xlabel("Período")
    ax.set_ylabel("Total de Ventas (S/)")

def grafico_productos_mas_vendidos(df: pd.DataFrame, ax: plt.Axes) -> None:
    datos = (df[df["producto"] != "Desconocido"]
             .groupby("producto")["cantidad"]
             .sum()
             .sort_values(ascending=False)
             .head(10)
             .reset_index())

    # Usamos Grafico de Barras
    sns.barplot(
        data=datos, 
        x="cantidad", 
        y="producto", 
        ax=ax, 
        palette="viridis", 
        hue="producto", 
        legend=False
    )

    for i, (_, fila) in enumerate(datos.iterrows()):
        ax.text(fila["cantidad"] + datos["cantidad"].max() * 0.01, i, f"{fila['cantidad']:,.0f}", va="center", fontsize=9)
        
    ax.set_title("Top 10 Productos más Vendidos", fontweight='bold')
    ax.set_xlabel("Unidades Vendidas")
    ax.set_ylabel("")

def grafico_ventas_por_categoria(df: pd.DataFrame, ax: plt.Axes) -> None:
    datos = (df[df["categoria"] != "desconocido"]
             .groupby("categoria")["total"]
             .sum()
             .sort_values(ascending=False)
             .reset_index())

    # Usamos Grafico de Barras
    sns.barplot(
        data=datos, 
        x="categoria", 
        y="total", 
        ax=ax, 
        palette="magma", 
        hue="categoria", 
        legend=False
    )

    for i, (_, fila) in enumerate(datos.iterrows()):
        ax.text(i, fila["total"] + datos["total"].max() * 0.01, f"S/ {fila['total']:,.0f}", ha="center", va="bottom", fontsize=8.5)
        
    ax.set_title("Ventas Totales por Categoría", fontweight='bold')
    ax.set_xlabel("Categoría")
    ax.set_ylabel("Ingresos Totales (S/)")
    ax.tick_params(axis="x", rotation=20)

def grafico_distribucion_precios(df: pd.DataFrame, ax: plt.Axes) -> None:
    precios = df["precio_unitario"].dropna()

    # Grafico Histograma
    sns.histplot(
        precios, 
        bins=20, 
        kde=True, 
        color="purple", 
        alpha=0.5, 
        ax=ax
    )

    media = precios.mean()
    mediana = precios.median()
    ax.axvline(media, color="red", linestyle="--", linewidth=1.5, label=f"Media S/ {media:.2f}")
    ax.axvline(mediana, color="green", linestyle=":", linewidth=1.5, label=f"Mediana S/ {mediana:.2f}")
    ax.legend(fontsize=9)

    ax.set_title("Distribución de Precios Unitarios", fontweight='bold')
    ax.set_xlabel("Precio Unitario (S/)")
    ax.set_ylabel("Frecuencia")

def grafico_correlacion(df: pd.DataFrame, ax: plt.Axes) -> None:
    columnas_numericas = ["cantidad", "precio_unitario", "descuento_pct", "satisfaccion", "total"]
    corr = df[columnas_numericas].dropna().corr()

    # Usamos el Grafico de Calor
    sns.heatmap(
        corr, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        center=0, 
        linewidths=0.5, 
        ax=ax
    )
    ax.set_title("Correlación entre Variables Numéricas", fontweight='bold')

def grafico_histograma_satisfaccion(df: pd.DataFrame, ax: plt.Axes) -> None:

    # Usamos histplot directo de Seaborn para cumplir estrictamente con el histograma
    sns.histplot(
        data=df, 
        x="satisfaccion", 
        discrete=True, 
        color="teal", 
        alpha=0.7, 
        ax=ax
    )
    ax.set_title("Histograma – Satisfacción del Cliente", fontweight='bold')
    ax.set_xlabel("Puntuación (1–5 estrellas)")
    ax.set_ylabel("Número de Transacciones")
    ax.set_xticks([1, 2, 3, 4, 5])


# 2. FUNCIÓN PRINCIPAL DEL DASHBOARD
# ------------------------------------------
def mostrar_dashboard(df: pd.DataFrame) -> None:
    print("=" * 52)
    print("        DASHBOARD ESTADÍSTICO")
    print("        Sistema Supermercado")
    print("=" * 52)
    print(f" Registros cargados : {len(df):,}")
    print(f" Ventas totales     : S/ {df['total'].sum():,.2f}")
    print("=" * 52)

    # Crear la cuadrícula de subplots 3x2
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 15), facecolor="#f9f9f9")
    fig.suptitle("Dashboard Estadístico – Sistema Supermercado", fontsize=18, fontweight="bold", y=0.98)

    # Inyectar datos en cada gráfico
    grafico_ventas_por_mes(df, axes[0, 0])
    grafico_productos_mas_vendidos(df, axes[0, 1])
    grafico_ventas_por_categoria(df, axes[1, 0])
    grafico_distribucion_precios(df, axes[1, 1])
    grafico_correlacion(df, axes[2, 0])
    grafico_histograma_satisfaccion(df, axes[2, 1])

    # Ajustar espacios para que no colisionen los textos
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# 3. PUNTO DE ENTRADA LOCAL
# ------------------------------------------
if __name__ == "__main__":
    # La lectura del CSV solo ocurre si ejecutas este archivo directamente
    ruta_csv = "limpieza_datos.csv" # Asegúrate de que este sea el nombre correcto de tu archivo
    
    try:
        df_prueba = pd.read_csv(ruta_csv)
        mostrar_dashboard(df_prueba)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{ruta_csv}'.")