import pandas as pd
import numpy as np

# ==========================================
# 1. CARGAR EL DATASET
# ==========================================

df = pd.read_csv(
    "supermercado.csv",
    encoding="utf-8"
)

print("========== DATOS ORIGINALES ==========")
print(df.head())

print("\nFilas:", df.shape[0])
print("Columnas:", df.shape[1])

print("\nTipos de datos:")
print(df.dtypes)


# ==========================================
# 2. DETECTAR DATOS FALTANTES
# ==========================================

print("\n========== DATOS FALTANTES ==========")
print(df.isnull().sum())


# ==========================================
# 3. ELIMINAR ESPACIOS EN TEXTOS
# ==========================================

for columna in df.select_dtypes(include="object").columns:
    df[columna] = df[columna].str.strip()


# ==========================================
# 4. CONVERTIR FECHA
# ==========================================

df["fecha"] = pd.to_datetime(
    df["fecha"],
    errors="coerce"
)


# ==========================================
# 5. ASIGNAR CATEGORÍA SEGÚN PRODUCTO
# ==========================================

mapa_categorias = {
    "Arroz": "Abarrotes",
    "Aceite": "Abarrotes",
    "Atún": "Abarrotes",
    "Azúcar": "Abarrotes",
    "Fideos": "Abarrotes",
    "Huevos": "Abarrotes",

    "Pan": "Panadería",
    "Leche": "Lácteos",
    "Gaseosa": "Bebidas",
    "Detergente": "Limpieza"
}

df["categoria"] = df["producto"].map(mapa_categorias)

# Si el producto es desconocido,
# la categoría también será desconocida

df["producto"] = df["producto"].fillna("Desconocido")
df["categoria"] = df["categoria"].fillna("Desconocido")


# ==========================================
# 6. CONVERTIR COLUMNAS NUMÉRICAS
# ==========================================

columnas_numericas = [
    "cantidad",
    "precio_unitario",
    "descuento_pct",
    "satisfaccion",
    "total"
]

for columna in columnas_numericas:

    df[columna] = (
        df[columna]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )

    df[columna] = pd.to_numeric(
        df[columna],
        errors="coerce"
    )


# ==========================================
# 7. CORREGIR VALORES INVÁLIDOS
# ==========================================

# Cantidad debe ser positiva
df.loc[
    df["cantidad"] <= 0,
    "cantidad"
] = np.nan


# Precio debe ser positivo
df.loc[
    df["precio_unitario"] <= 0,
    "precio_unitario"
] = np.nan


# Descuento válido entre 0 y 100
df.loc[
    (df["descuento_pct"] < 0) |
    (df["descuento_pct"] > 100),
    "descuento_pct"
] = np.nan


# Satisfacción válida entre 1 y 5
df.loc[
    (df["satisfaccion"] < 1) |
    (df["satisfaccion"] > 5),
    "satisfaccion"
] = np.nan


# Total no puede ser negativo
df.loc[
    df["total"] < 0,
    "total"
] = np.nan


# ==========================================
# 8. ELIMINAR FILAS IMPORTANTES INCOMPLETAS
# ==========================================

# Sin ID o fecha no podemos identificar
# correctamente una venta

df = df.dropna(
    subset=[
        "id_venta",
        "fecha"
    ]
)

df["id_venta"] = df["id_venta"].astype(int)


# ==========================================
# 9. NORMALIZAR TEXTO
# ==========================================

# Producto:
# primera letra de cada palabra en mayúscula

df["producto"] = (
    df["producto"]
    .str.lower()
    .str.title()
)


# Categoría en minúsculas
df["categoria"] = (
    df["categoria"]
    .str.lower()
)


# Método de pago en minúsculas
df["metodo_pago"] = (
    df["metodo_pago"]
    .str.lower()
)


# Cliente frecuente en minúsculas
df["cliente_frecuente"] = (
    df["cliente_frecuente"]
    .str.lower()
)


# Opinión del usuario en minúsculas
df["opinion_usuario"] = (
    df["opinion_usuario"]
    .str.lower()
)


# ==========================================
# 10. TRATAR DATOS FALTANTES
# ==========================================

# Para cantidad y precio utilizamos la mediana

for columna in [
    "cantidad",
    "precio_unitario"
]:

    df[columna] = df[columna].fillna(
        df[columna].median()
    )


# Para descuento asumimos 0 si falta

df["descuento_pct"] = (
    df["descuento_pct"]
    .fillna(0)
)


# Para satisfacción utilizamos la mediana

df["satisfaccion"] = (
    df["satisfaccion"]
    .fillna(
        df["satisfaccion"].median()
    )
)


# Datos categóricos faltantes

columnas_texto = [
    "producto",
    "categoria",
    "metodo_pago",
    "cliente_frecuente",
    "opinion_usuario"
]

for columna in columnas_texto:

    df[columna] = df[columna].fillna(
        "desconocido"
    )


# ==========================================
# 11. CALCULAR / CORREGIR TOTAL
# ==========================================

# Calculamos cuánto debería valer
# cada venta

total_calculado = (
    df["cantidad"]
    * df["precio_unitario"]
    * (
        1 -
        df["descuento_pct"] / 100
    )
)


# Si el total está vacío,
# utilizamos el total calculado

df["total"] = df["total"].fillna(
    total_calculado
)


# ==========================================
# 12. ELIMINAR DUPLICADOS
# ==========================================

df = df.drop_duplicates()


# ==========================================
# 13. CREAR NUEVAS COLUMNAS
# ==========================================

df["año"] = (
    df["fecha"].dt.year
)

df["mes"] = (
    df["fecha"].dt.month
)

df["nombre_mes"] = (
    df["fecha"].dt.month_name()
)

df["año_mes"] = (
    df["fecha"]
    .dt.to_period("M")
    .astype(str)
)


# ==========================================
# 14. MÉTRICAS GENERALES
# ==========================================

ventas_totales = (
    df["total"].sum()
)

venta_promedio = (
    df["total"].mean()
)

cantidad_productos = (
    df["cantidad"].sum()
)

satisfaccion_promedio = (
    df["satisfaccion"].mean()
)


print("\n========== MÉTRICAS GENERALES ==========")

print(
    "Ventas totales:",
    round(ventas_totales, 2)
)

print(
    "Venta promedio:",
    round(venta_promedio, 2)
)

print(
    "Cantidad total de productos vendidos:",
    round(cantidad_productos, 0)
)

print(
    "Satisfacción promedio:",
    round(satisfaccion_promedio, 2)
)


# ==========================================
# 15. PRODUCTO MÁS VENDIDO
# ==========================================

ventas_producto = (
    df.groupby("producto")["cantidad"]
    .sum()
    .sort_values(
        ascending=False
    )
)

print(
    "\n========== PRODUCTOS MÁS VENDIDOS =========="
)

print(ventas_producto)

print(
    "\nProducto más vendido:",
    ventas_producto.index[0]
)

print(
    "Unidades vendidas:",
    ventas_producto.iloc[0]
)


# ==========================================
# 16. CATEGORÍA CON MAYORES INGRESOS
# ==========================================

ventas_categoria = (
    df.groupby("categoria")["total"]
    .sum()
    .sort_values(
        ascending=False
    )
)

print(
    "\n========== VENTAS POR CATEGORÍA =========="
)

print(ventas_categoria)

print(
    "\nCategoría con mayores ingresos:",
    ventas_categoria.index[0]
)

print(
    "Ingresos:",
    round(
        ventas_categoria.iloc[0],
        2
    )
)


# ==========================================
# 17. VENTAS POR MES
# ==========================================

ventas_mes = (
    df.groupby("año_mes")["total"]
    .sum()
    .sort_index()
)

print(
    "\n========== VENTAS POR MES =========="
)

print(ventas_mes)


# ==========================================
# 18. MÉTODOS DE PAGO
# ==========================================

metodos_pago = (
    df["metodo_pago"]
    .value_counts()
)

print(
    "\n========== MÉTODOS DE PAGO =========="
)

print(metodos_pago)


# ==========================================
# 19. CLIENTES FRECUENTES
# ==========================================

clientes = (
    df["cliente_frecuente"]
    .value_counts()
)

print(
    "\n========== CLIENTES FRECUENTES =========="
)

print(clientes)


# ==========================================
# 20. COMPROBAR DATASET LIMPIO
# ==========================================

print(
    "\n========== DATASET LIMPIO =========="
)

print(df.head())

print(
    "\nDatos faltantes después de limpiar:"
)

print(df.isnull().sum())

print(
    "\nCantidad final de filas:",
    len(df)
)

print(
    "\nCantidad final de columnas:",
    len(df.columns)
)


# ==========================================
# 21. GUARDAR DATASET LIMPIO
# ==========================================

df.to_csv(
    "supermercado_final20.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\n========== PROCESO TERMINADO =========="
)

print(
    "Archivo creado: supermercado_final20.csv"
)