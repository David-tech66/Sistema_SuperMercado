import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ==========================================
# 1. CARGAR EL DATASET
# ==========================================

df = pd.read_csv("supermercado_final20.csv")

print("========================================")
print("       MÓDULO 3 - INTELIGENCIA")
print("          SOBRE OPINIONES")
print("========================================")

print("Registros originales:", len(df))


# ==========================================
# 2. ELIMINAR DUPLICADOS
# ==========================================

df = df.drop_duplicates()

print(
    "Registros después de eliminar duplicados:",
    len(df)
)


# ==========================================
# 3. PREPARAR LAS OPINIONES
# ==========================================

df["opinion_usuario"] = df[
    "opinion_usuario"
].astype(str)

df["opinion_usuario"] = df[
    "opinion_usuario"
].str.lower()


# ==========================================
# 4. ELIMINAR OPINIONES INVÁLIDAS
# ==========================================

invalidas = [
    "desconocido",
    "sin comentario",
    "???",
    "0"
]

df = df[
    ~df["opinion_usuario"].isin(invalidas)
]

print(
    "Opiniones válidas:",
    len(df)
)


# ==========================================
# 5. NLTK
# ==========================================

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

stop_words = stopwords.words("spanish")

# Conservamos "no" porque puede indicar
# una opinión negativa.

if "no" in stop_words:
    stop_words.remove("no")


# ==========================================
# 6. LIMPIAR EL TEXTO
# ==========================================

def limpiar_texto(texto):

    # Eliminamos símbolos y números
    texto = re.sub(
        r"[^a-záéíóúñü\s]",
        "",
        texto
    )

    # Separamos las palabras
    palabras = word_tokenize(
        texto,
        language="spanish"
    )

    # Eliminamos stopwords
    palabras = [
        palabra
        for palabra in palabras
        if palabra not in stop_words
    ]

    # Volvemos a unir las palabras
    return " ".join(palabras)


df["opinion_limpia"] = df[
    "opinion_usuario"
].apply(limpiar_texto)


# ==========================================
# 7. CLASIFICACIÓN DE OPINIONES ORIGINALES
# ==========================================

positivos = [
    "buenos precios y productos frescos",
    "la compra fue rápida y ordenada",
    "ofertas interesantes y personal amable",
    "muy buena variedad de productos",
    "excelente atención y encontré todo lo que necesitaba",
    "excelente!!!",
    "todo bien"
]

negativos = [
    "había productos vencidos en el estante",
    "faltaban varios productos básicos",
    "los precios no coincidían con la etiqueta",
    "mala atención del personal",
    "mucha cola para pagar"
]

positivos = [limpiar_texto(opinion) for opinion in positivos]
negativos = [limpiar_texto(opinion) for opinion in negativos]

def clasificar_sentimiento(opinion):

    opinion = limpiar_texto(opinion)

    if opinion in positivos:
        return "POSITIVE"

    elif opinion in negativos:
        return "NEGATIVE"

    else:
        return None


df["sentimiento"] = df[
    "opinion_usuario"
].apply(clasificar_sentimiento)


# Eliminamos opiniones que no pudieron
# recibir una etiqueta.

df = df.dropna(
    subset=["sentimiento"]
)


print("\n========================================")
print("       CLASIFICACIÓN DE OPINIONES")
print("========================================")

print(
    "Opiniones clasificadas:",
    len(df)
)

print("\nCantidad por sentimiento:")

print(
    df["sentimiento"].value_counts()
)


# ==========================================
# 8. EJEMPLOS ADICIONALES
# ==========================================

positivas_extra = [

    "recomiendo este supermercado",
    "los recomiendo",
    "recomiendo comprar aquí",
    "muy buena experiencia",
    "la atención fue excelente",
    "excelente atención del personal",
    "hay muchos productos disponibles",
    "hay suficientes productos",
    "los productos están frescos",
    "los precios son buenos",
    "la compra fue rápida",
    "encontré todo lo que buscaba",
    "encontré todo lo que necesitaba",
    "volvería a comprar aquí",
    "me gustó la atención",
    "estoy satisfecho con la compra",
    "me gustó mucho la variedad",
    "todo estuvo muy bien",
    "la experiencia fue muy buena",
    "compraría nuevamente aquí",

    "me gustó mucho comprar aquí",
    "la atención fue muy buena",
    "los productos tienen buena calidad",
    "encontré buenos productos",
    "los precios son económicos",
    "la compra fue excelente",
    "volvería nuevamente",
    "estoy muy satisfecho",
    "recomiendo totalmente este lugar",
    "todo estuvo excelente",
    "la variedad es muy buena",
    "me atendieron muy bien",
    "encontré todo fácilmente",
    "la experiencia fue excelente",
    "compraría otra vez aquí"
]


negativas_extra = [

    "no recomiendo este supermercado",
    "no los recomiendo",
    "no recomiendo comprar aquí",
    "no volvería a comprar aquí",
    "no volvería a este supermercado",
    "muy mala experiencia",
    "la atención fue pésima",
    "muy mala atención",
    "mala atención del personal",
    "no me gustó la atención",
    "no estoy satisfecho con la compra",
    "faltan productos básicos",
    "faltan muchos productos",
    "faltan productos de primera necesidad",
    "no encontré lo que buscaba",
    "no encontré los productos que buscaba",
    "los productos están vencidos",
    "los precios son demasiado altos",
    "demasiada cola para pagar",
    "la experiencia fue muy mala",

    "no los recomiendo porque faltan productos",
    "no recomiendo esta tienda porque faltan productos",
    "no volvería porque la atención fue mala",
    "no estoy satisfecho porque la atención fue pésima",
    "no recomiendo el supermercado porque hay pocos productos",
    "no encontré productos de primera necesidad",
    "faltan muchos productos básicos",
    "la atención fue terrible",
    "la atención fue muy mala",
    "los precios son demasiado altos y no recomiendo la tienda",
    "no volvería a comprar porque faltan productos",
    "no me gustó nada la atención",
    "la experiencia fue terrible",
    "no recomiendo este lugar",
    "no compraría nuevamente aquí"
]


# ==========================================
# 9. CREAR DATOS ADICIONALES
# ==========================================

opiniones_extra = (

    [
        (opinion, "POSITIVE")
        for opinion in positivas_extra
    ]

    +

    [
        (opinion, "NEGATIVE")
        for opinion in negativas_extra
    ]
)


df_extra = pd.DataFrame(
    opiniones_extra,
    columns=[
        "opinion_usuario",
        "sentimiento"
    ]
)


df_extra["opinion_limpia"] = df_extra[
    "opinion_usuario"
].apply(limpiar_texto)


# ==========================================
# 10. UNIR LOS DATOS
# ==========================================

df = pd.concat(
    [df, df_extra],
    ignore_index=True
)


print("\n========================================")
print("          DATOS PARA EL MODELO")
print("========================================")

print(
    "Opiniones originales:",
    len(df) - len(df_extra)
)

print(
    "Ejemplos adicionales:",
    len(df_extra)
)

print(
    "Total de opiniones:",
    len(df)
)

print("\nOpiniones por sentimiento:")

print(
    df["sentimiento"].value_counts()
)


# ==========================================
# 11. SEPARAR DATOS
# ==========================================

X = df["opinion_limpia"]

y = df["sentimiento"]


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\n========================================")
print("          DIVISIÓN DE DATOS")
print("========================================")

print(
    "Datos para entrenamiento:",
    len(X_train)
)

print(
    "Datos para prueba:",
    len(X_test)
)


# ==========================================
# 12. TF-IDF
# ==========================================

vectorizador = TfidfVectorizer(
    ngram_range=(1, 2)
)


X_train = vectorizador.fit_transform(
    X_train
)

X_test = vectorizador.transform(
    X_test
)


# ==========================================
# 13. REGRESIÓN LOGÍSTICA
# ==========================================

modelo = LogisticRegression(
    max_iter=1000
)


modelo.fit(
    X_train,
    y_train
)


print(
    "\nModelo entrenado correctamente."
)


# ==========================================
# 14. EVALUAR EL MODELO
# ==========================================

predicciones = modelo.predict(
    X_test
)


precision = accuracy_score(
    y_test,
    predicciones
)


print("\n========================================")
print("        RESULTADO DEL MODELO")
print("========================================")

print(
    "Precisión:",
    round(
        precision * 100,
        2
    ),
    "%"
)


# ==========================================
# 15. NUEVA OPINIÓN
# ==========================================

opinion = input(
    "\nEscribe una opinión: "
)


# Limpiamos la opinión

opinion_limpia = limpiar_texto(
    opinion
)


# Convertimos la opinión a números

opinion_numerica = vectorizador.transform(
    [opinion_limpia]
)


# ==========================================
# 16. PREDICCIÓN
# ==========================================

resultado = modelo.predict(
    opinion_numerica
)[0]


# ==========================================
# 17. PROBABILIDAD
# ==========================================

probabilidades = modelo.predict_proba(
    opinion_numerica
)[0]


indice_resultado = list(
    modelo.classes_
).index(resultado)


probabilidad = (
    probabilidades[indice_resultado]
    * 100
)


# ==========================================
# 18. MOSTRAR RESULTADO
# ==========================================

print("\n========================================")
print("             RESULTADO")
print("========================================")

print(
    "Opinión:",
    opinion
)

print(
    "Sentimiento:",
    resultado
)

print(
    "Probabilidad:",
    round(
        probabilidad,
        2
    ),
    "%"
)