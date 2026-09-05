import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import warnings
warnings.filterwarnings("ignore")


# ==========================================
# CONFIGURACIÓN GLOBAL DE MODELOS
# ==========================================

MODELOS = {
    "Regresión Lineal": LinearRegression(),
    "Random Forest":    RandomForestRegressor(
                            n_estimators=100,
                            random_state=42
                        ),
    "Decision Tree":    DecisionTreeRegressor(
                            max_depth=8,
                            random_state=42
                        ),
}

# Encoders globales para reutilizarlos en predicciones
_encoders = {}

# Modelo seleccionado como el mejor (se rellena al entrenar)
_mejor_modelo_nombre = None
_mejor_modelo_objeto = None
_columnas_entrenamiento = None


# ==========================================
# 1. CARGAR DATOS
# ==========================================

def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    """
    Carga el dataset limpio.
    Busca automáticamente 'supermercado_limpio.csv'
    en rutas relativas al módulo si no se indica ruta.
    """
    if ruta_csv is None:
        candidatas = [
            "supermercado_limpio.csv",
            "../supermercado_limpio.csv",
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "supermercado_limpio.csv"
            ),
        ]
        for ruta in candidatas:
            if os.path.exists(ruta):
                ruta_csv = ruta
                break

    if ruta_csv is None or not os.path.exists(ruta_csv):
        raise FileNotFoundError(
            "No se encontró 'supermercado_limpio.csv'. "
            "Ejecuta primero supermercado2.py para generarlo."
        )

    df = pd.read_csv(ruta_csv, encoding="utf-8-sig")
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    return df


# ==========================================
# 2. INGENIERÍA DE FEATURES
# ==========================================

def preparar_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Construye la matriz de features X y el vector objetivo y
    a partir del dataset limpio.

    Features utilizadas
    -------------------
    - producto          → codificado como entero
    - categoria         → codificada como entero
    - dia_semana        → 0=Lunes … 6=Domingo
    - mes               → 1–12
    - año               → 2022, 2023, …
    - precio_unitario   → precio en S/
    - descuento_pct     → porcentaje de descuento (0–100)
    - cliente_frecuente → 0 (no) / 1 (si)
    - satisfaccion      → puntuación 1–5
    - ventas_hist_prod  → media histórica de 'cantidad' por producto
    - ventas_hist_mes   → media histórica de 'cantidad' por mes

    Objetivo (y)
    ------------
    - cantidad          → unidades vendidas en esa transacción
    """

    df = df.copy()

    # --- Variables temporales ---
    df["dia_semana"] = df["fecha"].dt.dayofweek   # 0=Lun … 6=Dom

    # --- Encoders para variables categóricas ---
    for col in ["producto", "categoria"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(
            df[col].astype(str)
        )
        _encoders[col] = le

    # --- Cliente frecuente como binario ---
    df["cliente_frecuente_bin"] = (
        df["cliente_frecuente"]
        .str.strip()
        .str.lower()
        .map({"si": 1, "sí": 1, "yes": 1})
        .fillna(0)
        .astype(int)
    )

    # --- Ventas históricas por producto (media) ---
    hist_prod = (
        df.groupby("producto")["cantidad"]
        .mean()
        .rename("ventas_hist_prod")
    )
    df = df.merge(hist_prod, on="producto", how="left")

    # --- Ventas históricas por mes ---
    hist_mes = (
        df.groupby("mes")["cantidad"]
        .mean()
        .rename("ventas_hist_mes")
    )
    df = df.merge(hist_mes, on="mes", how="left")

    # --- Selección final de columnas ---
    columnas_features = [
        "producto_enc",
        "categoria_enc",
        "dia_semana",
        "mes",
        "año",
        "precio_unitario",
        "descuento_pct",
        "cliente_frecuente_bin",
        "satisfaccion",
        "ventas_hist_prod",
        "ventas_hist_mes",
    ]

    X = df[columnas_features].fillna(0)
    y = df["cantidad"].fillna(0)

    return X, y


# ==========================================
# 3. ENTRENAR Y COMPARAR MODELOS
# ==========================================

def entrenar_y_comparar(
    df: pd.DataFrame,
    test_size: float = 0.2,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Entrena los 3 modelos, evalúa sus métricas y devuelve
    un DataFrame comparativo.  También deja el mejor modelo
    listo para hacer predicciones con `predecir_ventas()`.
    """

    global _mejor_modelo_nombre, _mejor_modelo_objeto
    global _columnas_entrenamiento

    X, y = preparar_features(df)
    _columnas_entrenamiento = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42
    )

    resultados = []

    for nombre, modelo in MODELOS.items():

        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        resultados.append({
            "Modelo":  nombre,
            "MAE":     round(mae,  4),
            "RMSE":    round(rmse, 4),
            "R²":      round(r2,   4),
        })

        if verbose:
            print(f"\n{'='*45}")
            print(f"  {nombre}")
            print(f"{'='*45}")
            print(f"  MAE  (Error Absoluto Medio)  : {mae:.4f}")
            print(f"  RMSE (Raíz Error Cuadrático) : {rmse:.4f}")
            print(f"  R²   (Coef. Determinación)   : {r2:.4f}")

    df_resultados = pd.DataFrame(resultados)

    # El mejor modelo tiene el R² más alto
    idx_mejor = df_resultados["R²"].idxmax()
    _mejor_modelo_nombre = df_resultados.loc[idx_mejor, "Modelo"]

    # Re-entrenamos el mejor con TODOS los datos para maximizar
    # la información disponible al predecir
    mejor_modelo_base = MODELOS[_mejor_modelo_nombre]
    mejor_modelo_base.fit(X, y)
    _mejor_modelo_objeto = mejor_modelo_base

    if verbose:
        print(f"\n{'='*45}")
        print(f"  ✅ MEJOR MODELO: {_mejor_modelo_nombre}")
        print(f"     R² = {df_resultados.loc[idx_mejor, 'R²']}")
        print(f"{'='*45}\n")
        print("\nTabla comparativa completa:")
        print(df_resultados.to_string(index=False))

    return df_resultados


# ==========================================
# 4. FUNCIÓN DE PREDICCIÓN INDIVIDUAL
# ==========================================

def predecir_ventas(
    producto: str,
    mes: int,
    año: int,
    precio_unitario: float,
    descuento_pct: float,
    dia_semana: int = 0,
    cliente_frecuente: str = "no",
    satisfaccion: float = 3.0,
    df_referencia: pd.DataFrame = None,
) -> dict:
    """
    Predice la cantidad de unidades que se venderán de un producto.

    Parámetros
    ----------
    producto         : nombre del producto (ej. 'Arroz')
    mes              : número de mes (1–12)
    año              : año (ej. 2024)
    precio_unitario  : precio en S/ (ej. 25.50)
    descuento_pct    : porcentaje de descuento, 0–100 (ej. 10.0)
    dia_semana       : 0=Lunes … 6=Domingo (default 0)
    cliente_frecuente: 'si' o 'no' (default 'no')
    satisfaccion     : puntuación 1–5 (default 3.0)
    df_referencia    : DataFrame limpio para calcular promedios históricos.
                       Si es None, usa 0 como fallback.

    Retorna
    -------
    dict con 'prediccion', 'modelo_usado', 'detalles'
    """

    if _mejor_modelo_objeto is None:
        raise RuntimeError(
            "El modelo no ha sido entrenado. "
            "Llama primero a entrenar_y_comparar()."
        )

    producto_norm = producto.strip().lower().title()

    # Encoding de producto
    le_prod = _encoders.get("producto")
    if le_prod and producto_norm in le_prod.classes_:
        prod_enc = int(le_prod.transform([producto_norm])[0])
    else:
        # Producto nuevo / desconocido → promedio de clases
        prod_enc = len(le_prod.classes_) // 2 if le_prod else 0

    # Encoding de categoría (inferida del producto)
    mapa_cat = {
        "Arroz": "abarrotes", "Aceite": "abarrotes",
        "Atún":  "abarrotes", "Azúcar": "abarrotes",
        "Fideos": "abarrotes", "Huevos": "abarrotes",
        "Pan":   "panadería",  "Leche":  "lácteos",
        "Gaseosa": "bebidas",  "Detergente": "limpieza",
    }
    categoria = mapa_cat.get(producto_norm, "desconocido")

    le_cat = _encoders.get("categoria")
    if le_cat and categoria in le_cat.classes_:
        cat_enc = int(le_cat.transform([categoria])[0])
    else:
        cat_enc = len(le_cat.classes_) // 2 if le_cat else 0

    # Ventas históricas
    if df_referencia is not None:
        hist_prod = (
            df_referencia[
                df_referencia["producto"].str.lower().str.title()
                == producto_norm
            ]["cantidad"].mean()
        )
        hist_mes = (
            df_referencia[
                df_referencia["mes"] == mes
            ]["cantidad"].mean()
        )
        hist_prod = hist_prod if not pd.isna(hist_prod) else 0.0
        hist_mes  = hist_mes  if not pd.isna(hist_mes)  else 0.0
    else:
        hist_prod = 0.0
        hist_mes  = 0.0

    # Cliente frecuente binario
    frec_bin = (
        1 if str(cliente_frecuente).strip().lower()
        in ("si", "sí", "yes", "1")
        else 0
    )

    # Construir el vector de features en el mismo orden del entrenamiento
    fila = pd.DataFrame([{
        "producto_enc":       prod_enc,
        "categoria_enc":      cat_enc,
        "dia_semana":         dia_semana,
        "mes":                mes,
        "año":                año,
        "precio_unitario":    precio_unitario,
        "descuento_pct":      descuento_pct,
        "cliente_frecuente_bin": frec_bin,
        "satisfaccion":       satisfaccion,
        "ventas_hist_prod":   hist_prod,
        "ventas_hist_mes":    hist_mes,
    }])

    # Alinear columnas por si acaso
    fila = fila[_columnas_entrenamiento]

    prediccion_raw = _mejor_modelo_objeto.predict(fila)[0]
    prediccion     = max(0.0, round(float(prediccion_raw), 2))

    return {
        "prediccion":   prediccion,
        "unidad":       "unidades",
        "modelo_usado": _mejor_modelo_nombre,
        "detalles": {
            "producto":          producto_norm,
            "mes":               mes,
            "año":               año,
            "precio_unitario":   precio_unitario,
            "descuento_pct":     descuento_pct,
            "dia_semana":        dia_semana,
            "cliente_frecuente": cliente_frecuente,
            "satisfaccion":      satisfaccion,
            "hist_ventas_prod":  round(hist_prod, 2),
            "hist_ventas_mes":   round(hist_mes,  2),
        }
    }


# ==========================================
# 5. IMPORTANCIA DE VARIABLES (Random Forest)
# ==========================================

def mostrar_importancia_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Entrena Random Forest y devuelve la importancia de cada feature.
    Útil para entender qué variables impactan más en las ventas.
    """
    X, y = preparar_features(df)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importancias = pd.DataFrame({
        "Feature":    X.columns,
        "Importancia": rf.feature_importances_
    }).sort_values("Importancia", ascending=False).reset_index(drop=True)

    print("\n📊 Importancia de Variables (Random Forest):")
    print(importancias.to_string(index=False))

    return importancias


# ==========================================
# 6. PUNTO DE ENTRADA DIRECTO
# ==========================================

if __name__ == "__main__":

    print("=" * 55)
    print("  SISTEMA DE PREDICCIÓN DE VENTAS — SUPERMERCADO")
    print("=" * 55)

    # --- Cargar datos ---
    print("\n📂 Cargando dataset limpio...")
    df = cargar_datos()
    print(f"   Registros cargados: {len(df)}")
    print(f"   Columnas          : {list(df.columns)}")

    # --- Entrenar y comparar modelos ---
    print("\n🤖 Entrenando y comparando modelos...\n")
    tabla = entrenar_y_comparar(df, test_size=0.2, verbose=True)

    # --- Importancia de variables ---
    importancias = mostrar_importancia_variables(df)

    # --- Ejemplo de predicción ---
    print("\n" + "=" * 55)
    print("  EJEMPLO DE PREDICCIÓN")
    print("=" * 55)

    casos_prueba = [
        {
            "producto":        "Arroz",
            "mes":             3,
            "año":             2024,
            "precio_unitario": 28.50,
            "descuento_pct":   10.0,
            "dia_semana":      4,        # Viernes
            "cliente_frecuente": "si",
            "satisfaccion":    4.0,
        },
        {
            "producto":        "Gaseosa",
            "mes":             12,
            "año":             2024,
            "precio_unitario": 5.90,
            "descuento_pct":   0.0,
            "dia_semana":      6,        # Domingo
            "cliente_frecuente": "no",
            "satisfaccion":    3.0,
        },
        {
            "producto":        "Detergente",
            "mes":             7,
            "año":             2024,
            "precio_unitario": 12.00,
            "descuento_pct":   20.0,
            "dia_semana":      1,        # Martes
            "cliente_frecuente": "si",
            "satisfaccion":    5.0,
        },
    ]

    dias = [
        "Lunes", "Martes", "Miércoles",
        "Jueves", "Viernes", "Sábado", "Domingo"
    ]
    meses_nombre = [
        "", "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
    ]

    for i, caso in enumerate(casos_prueba, 1):
        resultado = predecir_ventas(**caso, df_referencia=df)

        print(f"\n  Caso {i}: {caso['producto']}")
        print(f"  {'─'*40}")
        print(f"  Mes          : {meses_nombre[caso['mes']]} {caso['año']}")
        print(f"  Día          : {dias[caso['dia_semana']]}")
        print(f"  Precio       : S/ {caso['precio_unitario']}")
        print(f"  Descuento    : {caso['descuento_pct']}%")
        print(f"  Cl. frecuente: {caso['cliente_frecuente']}")
        print(f"  Satisfacción : {caso['satisfaccion']}/5")
        print(f"  Hist. prod.  : {resultado['detalles']['hist_ventas_prod']} uds (promedio)")
        print(f"  Hist. mes    : {resultado['detalles']['hist_ventas_mes']} uds (promedio)")
        print(f"\n  ➤ PREDICCIÓN   : {resultado['prediccion']} unidades")
        print(f"  ➤ Modelo usado : {resultado['modelo_usado']}")

    # --- Tabla resumen final ---
    print("\n" + "=" * 55)
    print("  RESUMEN COMPARATIVO DE MODELOS")
    print("=" * 55)
    print(tabla.to_string(index=False))
    print()

    # Interpretación automática
    mejor = tabla.loc[tabla["R²"].idxmax()]
    print(f"\n  🏆 Mejor modelo: {mejor['Modelo']}")
    print(f"     • R²   = {mejor['R²']}  (más cercano a 1 es mejor)")
    print(f"     • MAE  = {mejor['MAE']}  (error promedio en unidades)")
    print(f"     • RMSE = {mejor['RMSE']}  (penaliza errores grandes)\n")

    # Interpretación de cada métrica
    print("  📌 Interpretación de métricas:")
    print("     MAE  → en promedio, el modelo se equivoca en X unidades")
    print("     RMSE → igual que MAE pero penaliza errores grandes más")
    print("     R²   → qué % de la variación en ventas explica el modelo")
    print("            (0 = no explica nada, 1 = explicación perfecta)")
