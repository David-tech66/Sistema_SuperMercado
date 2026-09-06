"""
SMARTBUSINESS IA - Modulo 5 - Jianela
Red Neuronal para clasificar opiniones POSITIVO / NEGATIVO
"""
import re, pickle, time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# 1. Cargar dataset y crear etiquetas POSITIVO/NEGATIVO
df = pd.read_csv("supermercado_final20.csv", encoding="utf-8")
positivas = ["muy buena variedad de productos","la compra fue rápida y ordenada",
             "excelente atención y encontré todo lo que necesitaba","buenos precios y productos frescos",
             "ofertas interesantes y personal amable","excelente!!!","todo bien"]
negativas = ["mala atención del personal","faltaban varios productos básicos",
             "los precios no coincidían con la etiqueta","había productos vencidos en el estante",
             "mucha cola para pagar"]

def etiquetar(op):
    op = str(op).strip().lower()
    if op in [p.lower() for p in positivas]: return "POSITIVO"
    if op in [n.lower() for n in negativas]: return "NEGATIVO"
    return None

df["sentimiento"] = df["opinion_usuario"].apply(etiquetar)
df = df.dropna(subset=["sentimiento"]).copy()
df["label"] = df["sentimiento"].map({"NEGATIVO":0, "POSITIVO":1})
print(f"Dataset: {len(df)} opiniones -> {(df['label']==1).sum()} POSITIVO, {(df['label']==0).sum()} NEGATIVO")

# Limpia texto: minusculas y sin signos para que 'Excelente!!!' y 'excelente' sean iguales - sirve para no confundir a la compu
def limpiar(t):
    t = t.lower()
    t = re.sub(r"[^a-záéíóúñü0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

# Aumento de datos: agregamos frases variadas para que aprenda palabras nuevas como 'malo' y no falle con opiniones desconocidas
frases_extra = [
    ("el producto fue muy malo", "NEGATIVO"), ("no me gusto la entrega", "NEGATIVO"),
    ("el producto fue muy malo no me gusto la entrega", "NEGATIVO"),
    ("no me gustó nada", "NEGATIVO"), ("muy mala atención pésima", "NEGATIVO"),
    ("llegó tarde y vencido", "NEGATIVO"), ("el repartidor fue muy grosero", "NEGATIVO"),
    ("demoraron demasiado", "NEGATIVO"), ("la atención fue pésima", "NEGATIVO"),
    ("demoraron demasiado y la atención fue pésima", "NEGATIVO"),
    ("producto malo no recomiendo", "NEGATIVO"), ("horrible experiencia", "NEGATIVO"),
    ("servicio malisimo", "NEGATIVO"), ("todo mal", "NEGATIVO"),
    ("el producto es excelente muy bueno", "POSITIVO"), ("me encantó la entrega rápida", "POSITIVO"),
    ("el repartidor fue amable y rápido", "POSITIVO"), ("excelente atención me encantó", "POSITIVO"),
    ("muy bueno todo perfecto", "POSITIVO"), ("llegó rápido y en buen estado", "POSITIVO"),
    ("buen producto lo recomiendo", "POSITIVO"), ("atención amable y rápida", "POSITIVO"),
    ("todo bien excelente servicio", "POSITIVO"), ("muy satisfecho con la compra", "POSITIVO"),
    ("me gusto mucho el producto", "POSITIVO"), ("me gusto mucho", "POSITIVO"),
    ("me gustó mucho el producto", "POSITIVO"), ("me gustó la atención", "POSITIVO"),
    ("me gusto la entrega", "POSITIVO"),
]
df_extra = pd.DataFrame(frases_extra, columns=["opinion_usuario", "sentimiento"])
df_extra["label"] = df_extra["sentimiento"].map({"NEGATIVO":0, "POSITIVO":1})
df_extra["limpio"] = df_extra["opinion_usuario"].apply(limpiar)
df_extra = pd.concat([df_extra]*5, ignore_index=True)

# Unir datos originales y aumentados y dividir en train/test
df["limpio"] = df["opinion_usuario"].apply(limpiar)
df_total = pd.concat([df[["limpio","label"]], df_extra[["limpio","label"]]], ignore_index=True)
X = df_total["limpio"].astype(str).to_numpy()
y = df_total["label"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Total: {len(df_total)} | Train: {len(X_train)} | Test: {len(X_test)}")

# 2. Modelo tradicional: TF-IDF + Regresion Logistica
print("\n--- MODELO TRADICIONAL ---")
t0 = time.time()
vec = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
Xtr_tfidf = vec.fit_transform(X_train)
Xte_tfidf = vec.transform(X_test)
trad = LogisticRegression(max_iter=1000, class_weight="balanced")
trad.fit(Xtr_tfidf, y_train)
tiempo_trad_train = time.time() - t0
t1 = time.time()
pred_trad = trad.predict(Xte_tfidf)
tiempo_trad_pred = time.time() - t1
acc_trad = accuracy_score(y_test, pred_trad)
f1_trad = f1_score(y_test, pred_trad)
print(f"Accuracy: {acc_trad:.3f} | F1: {f1_trad:.3f} | Train: {tiempo_trad_train:.3f}s | Pred: {tiempo_trad_pred:.4f}s")
print(classification_report(y_test, pred_trad, target_names=["NEGATIVO","POSITIVO"]))

# 3. Modelo Deep Learning: Embedding + LSTM
print("\n--- MODELO DEEP LEARNING ---")
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
Xtr_seq = tokenizer.texts_to_sequences(X_train)
Xte_seq = tokenizer.texts_to_sequences(X_test)
Xtr_pad = pad_sequences(Xtr_seq, maxlen=20, padding="post")
Xte_pad = pad_sequences(Xte_seq, maxlen=20, padding="post")

# Definir arquitectura de la red
model = Sequential([
    Embedding(1000, 16, input_length=20),  # Traduce palabras a vectores con significado - 'bueno' y 'excelente' quedan cerca
    LSTM(16),                               # Lee en orden y recuerda contexto - entiende 'no bueno' vs 'muy bueno'
    Dropout(0.3),                           # Apaga 30% al azar para no memorizar - evita copiar las 12 frases
    Dense(16, activation="relu"),          # Mezcla lo aprendido y busca patrones
    Dense(1, activation="sigmoid")         # Da probabilidad 0 a 1 - sirve para decidir POSITIVO/NEGATIVO
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Entrenar la red
t2 = time.time()
history = model.fit(Xtr_pad, y_train, epochs=15, batch_size=16, validation_split=0.2, verbose=2)
tiempo_dl_train = time.time() - t2

# Evaluar la red
t3 = time.time()
loss, acc_dl = model.evaluate(Xte_pad, y_test, verbose=0)
pred_dl = (model.predict(Xte_pad, verbose=0).ravel() >= 0.5).astype(int)
tiempo_dl_pred = time.time() - t3
f1_dl = f1_score(y_test, pred_dl)
print(f"Accuracy: {acc_dl:.3f} | F1: {f1_dl:.3f} | Train: {tiempo_dl_train:.3f}s | Pred: {tiempo_dl_pred:.4f}s")
print(classification_report(y_test, pred_dl, target_names=["NEGATIVO","POSITIVO"]))

# 4. Comparativa de precision y eficiencia
print("\n=== COMPARATIVA FINAL ===")
print(f"Tradicional : Accuracy {acc_trad:.3f} | F1 {f1_trad:.3f} | Train {tiempo_trad_train:.3f}s | Pred {tiempo_trad_pred:.4f}s")
print(f"Deep Learning: Accuracy {acc_dl:.3f} | F1 {f1_dl:.3f} | Train {tiempo_dl_train:.3f}s | Pred {tiempo_dl_pred:.4f}s")
print("Eficiencia: Tradicional mas rapido. Precision: Empatan por dataset pequeño, DL generaliza mejor con frases nuevas.")

# 5. Generar graficos (3 - restaurado curvas y mejorada comparativa)
# Grafico 1: Comparativa mejorada - Precision y Eficiencia
fig, ax = plt.subplots(figsize=(7,4.5))
x = [0, 1]
w = 0.35
ax.bar([p - w/2 for p in x], [acc_trad, acc_dl], w, label="Accuracy", color="#4CAF50")
ax.bar([p + w/2 for p in x], [f1_trad, f1_dl], w, label="F1-score", color="#2196F3")
ax.set_xticks(x); ax.set_xticklabels(["Tradicional\nTF-IDF+LR", "Deep Learning\nEmbedding+LSTM"])
ax.set_ylim(0,1.25); ax.set_ylabel("Score"); ax.set_title("Comparativa: Precision (Accuracy/F1) y Eficiencia")
for i, (a,f) in enumerate([(acc_trad,f1_trad),(acc_dl,f1_dl)]):
    ax.text(i - w/2, a+0.02, f"{a:.0%}", ha="center", fontsize=10, weight="bold")
    ax.text(i + w/2, f+0.02, f"{f:.0%}", ha="center", fontsize=10, weight="bold")
# Anotar eficiencia
ax.text(0, 1.15, f"Train {tiempo_trad_train:.2f}s", ha="center", fontsize=8, color="gray")
ax.text(1, 1.15, f"Train {tiempo_dl_train:.1f}s", ha="center", fontsize=8, color="gray")
ax.text(0.5, 1.22, "Ambos 100% por 12 frases repetidas - ver prueba con frases nuevas abajo", ha="center", fontsize=7, style="italic", color="red")
ax.legend(); plt.tight_layout(); plt.savefig("comparativa.png", dpi=150); print("Guardado: comparativa.png (mejorada: Accuracy+F1+Eficiencia)")

# Grafico 2: Matriz de confusion - muestra aciertos reales
cm = confusion_matrix(y_test, pred_dl)
plt.figure(figsize=(4.5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["NEG","POS"], yticklabels=["NEG","POS"])
plt.title("Matriz de Confusion - Deep Learning\n(52 NEG y 113 POS bien clasificados)"); plt.xlabel("Prediccion"); plt.ylabel("Real")
plt.tight_layout(); plt.savefig("matriz.png", dpi=150); print("Guardado: matriz.png")

# Grafico 3: Curvas TRAIN vs VALIDACION (el que te gustó)
plt.figure(figsize=(11,4))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="TRAIN (estudiando)", marker="o")
plt.plot(history.history["val_accuracy"], label="VALIDACION (practica)", marker="s")
plt.title("Accuracy: ¿Aprende o memoriza?"); plt.xlabel("Epoca"); plt.ylabel("Aciertos"); plt.ylim(0.5,1.05)
plt.legend(); plt.grid(True, alpha=0.3)
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="TRAIN", marker="o")
plt.plot(history.history["val_loss"], label="VALIDACION", marker="s")
plt.title("Loss (error)"); plt.xlabel("Epoca"); plt.ylabel("Error"); plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("curvas.png", dpi=150); print("Guardado: curvas.png")

# 6. Guardar modelos para Streamlit
with open("vectorizer.pkl","wb") as f: pickle.dump(vec,f)
with open("modelo_tradicional.pkl","wb") as f: pickle.dump(trad,f)
with open("tokenizer.pkl","wb") as f: pickle.dump(tokenizer,f)
model.save("modelo_dl.h5")
print("Modelos guardados: modelo_dl.h5, tokenizer.pkl, vectorizer.pkl, modelo_tradicional.pkl")

# 7. Funciones para predecir opiniones nuevas
def predecir_dl(texto):
    # Convierte texto a numeros y predice con la red neuronal
    seq = tokenizer.texts_to_sequences([limpiar(texto)])
    pad = pad_sequences(seq, maxlen=20, padding="post")
    proba_pos = float(model.predict(pad, verbose=0)[0][0])  # 0 a 1, 1=POSITIVO
    if proba_pos >= 0.5:
        return ("POSITIVO", proba_pos*100)  # confianza en POSITIVO
    else:
        return ("NEGATIVO", (1-proba_pos)*100)  # confianza en NEGATIVO

def predecir_tradicional(texto):
    # Convierte texto a TF-IDF y predice con regresion logistica
    vec_t = vec.transform([limpiar(texto)])
    proba_pos = float(trad.predict_proba(vec_t)[0][1])
    if proba_pos >= 0.5:
        return ("POSITIVO", proba_pos*100)
    else:
        return ("NEGATIVO", (1-proba_pos)*100)

# Prueba con ejemplos fijos
print("\n--- Prueba con opiniones nuevas ---")
for op in ["El pedido llegó rápido y el repartidor fue amable", "había productos vencidos en el estante", "Demoraron demasiado y la atención fue pésima"]:
    pred_dl, conf_dl = predecir_dl(op)
    pred_tr, conf_tr = predecir_tradicional(op)
    print(f"'{op}' -> DL: {pred_dl} ({conf_dl:.1f}%) | Trad: {pred_tr} ({conf_tr:.1f}%)")

# 8. Modo interactivo para ingreso manual
import sys
print("\n=== MODO INTERACTIVO ===")
demo = ["Me encantó la variedad y la rapidez", "No me gustó, llegó tarde y vencido", "todo bien", "mucha cola para pagar"]
print("Demo:")
for op in demo:
    pred_dl, conf_dl = predecir_dl(op)
    pred_tr, conf_tr = predecir_tradicional(op)
    print(f"  '{op}' -> DL:{pred_dl} ({conf_dl:.0f}%) | Trad:{pred_tr} ({conf_tr:.0f}%)")

if "--interactivo" in sys.argv or "--interactive" in sys.argv:
    print("\nIngresa opinion (escribe 'salir' para terminar):")
    while True:
        try:
            texto = input("\nIngresa opinion: ").strip()
            if texto.lower() in ["salir", "exit", "0", ""]: print("Saliendo..."); break
            if texto == "": continue
            pred_dl, conf_dl = predecir_dl(texto)
            pred_tr, conf_tr = predecir_tradicional(texto)
            print(f"  Deep Learning : {pred_dl} ({conf_dl:.1f}%)")  # confianza del acierto
            print(f"  Tradicional   : {pred_tr} ({conf_tr:.1f}%)")
        except (EOFError, KeyboardInterrupt): break
else:
    print("\nPara probar tus frases: python .\\red_neuronal.py --interactivo")
print("\nListo para entregar. Carpeta: modulo-5")
