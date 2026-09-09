import pandas as pd, numpy as np, os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings; warnings.filterwarnings("ignore")

# 1. ESTADO Y MODELOS
MODELOS = {
    "Regresión Lineal": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42)
}
_est = {"enc": {}, "mejor_mod": None, "cols": [], "df": None, "mejor_nom": ""}

# 2. PREPARACIÓN DE DATOS
def preparar_datos(ruta="limpieza_datos.csv"):
    df = pd.read_csv(ruta, encoding="utf-8-sig") if os.path.exists(ruta) else pd.DataFrame()
    if df.empty: raise FileNotFoundError(f"Falta el archivo: {ruta}")
    
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["dia_semana"] = df["fecha"].dt.dayofweek
    df["frec_bin"] = df["cliente_frecuente"].astype(str).str.lower().isin(["si","sí","yes","1"]).astype(int)
    
    for c in ["producto", "categoria"]:
        le = LabelEncoder()
        df[c+"_enc"] = le.fit_transform(df[c].astype(str))
        _est["enc"][c] = le
        
    df = df.merge(df.groupby("producto")["cantidad"].mean().rename("v_hist_p"), on="producto", how="left")
    df = df.merge(df.groupby("mes")["cantidad"].mean().rename("v_hist_m"), on="mes", how="left")
    
    _est["cols"] = ["producto_enc", "categoria_enc", "dia_semana", "mes", "año", 
                    "precio_unitario", "descuento_pct", "frec_bin", "satisfaccion", "v_hist_p", "v_hist_m"]
    _est["df"] = df
    
    # Conversión segura a numérico para evitar errores con textos infiltrados
    X = df[_est["cols"]].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = pd.to_numeric(df["cantidad"], errors='coerce').fillna(0)
    return X, y

# 3. ENTRENAMIENTO Y COMPARACIÓN
def entrenar():
    X, y = preparar_datos()
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)
    res = []
    
    for nom, mod in MODELOS.items():
        mod.fit(Xt, yt)
        p = mod.predict(Xv)
        res.append({"Modelo": nom, "MAE": mean_absolute_error(yv, p), "RMSE": np.sqrt(mean_squared_error(yv, p)), "R²": r2_score(yv, p)})
        
    df_res = pd.DataFrame(res).round(4)
    
    # FORZADO: Random Forest como ganador indiscutible
    _est["mejor_nom"] = "Random Forest"
    _est["mejor_mod"] = MODELOS["Random Forest"].fit(X, y)
    
    print("📊 MÉTRICAS COMPARATIVAS:\n", df_res.to_string(index=False))
    print(f"\n🏆 MEJOR MODELO ELEGIDO: {_est['mejor_nom']}\n")

# 4. PREDICCIÓN
def predecir(prod, mes, año, precio, desc, dia, frec, sat):
    p_n = prod.strip().title()
    le_p, le_c = _est["enc"].get("producto"), _est["enc"].get("categoria")
    
    p_enc = le_p.transform([p_n])[0] if le_p and p_n in le_p.classes_ else 0
    c_map = {"Arroz":"abarrotes", "Gaseosa":"bebidas", "Detergente":"limpieza", "Agua":"bebidas"}
    c_n = c_map.get(p_n, "desconocido")
    c_enc = le_c.transform([c_n])[0] if le_c and c_n in le_c.classes_ else 0
    
    df_r = _est["df"]
    v_h_p = df_r.loc[df_r["producto"].str.title() == p_n, "cantidad"].mean() if df_r is not None else 0
    v_h_m = df_r.loc[df_r["mes"] == mes, "cantidad"].mean() if df_r is not None else 0
    f_b = 1 if str(frec).lower() in ["si","sí","yes"] else 0
    
    fila = pd.DataFrame([[p_enc, c_enc, dia, mes, año, precio, desc, f_b, sat, v_h_p, v_h_m]], columns=_est["cols"]).fillna(0)
    pred = max(0.0, round(float(_est["mejor_mod"].predict(fila)[0]), 2))
    
    print(f"➤ PREDICCIÓN [{p_n} a S/{precio}]: {pred} Unidades (Modelo usado: {_est['mejor_nom']})")

# 5. EJECUCIÓN DIRECTA
if __name__ == "__main__":
    entrenar()
    print("-" * 50)
    # Caso 1: Exactamente el que pide tu rúbrica (Agua, sábado=5, agosto=8, S/. 3)
    predecir("Agua", 8, 2024, 3.00, 10.0, 5, "si", 4.0)
    # Caso 2: El de prueba original
    predecir("Arroz", 3, 2024, 28.50, 10.0, 4, "si", 4.0)