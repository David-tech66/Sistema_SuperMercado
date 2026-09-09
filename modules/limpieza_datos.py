import pandas as pd
import numpy as np
import csv
from io import StringIO

# 1. CARGAR DATASET ORIGINAL
with open("supermercado.csv", "r", encoding="utf-8-sig") as archivo:
    lineas = archivo.readlines()

filas = []
for linea in lineas:
    linea = linea.rstrip("\n\r")
    # Algunas filas pueden estar encerradas incorrectamente
    # entre comillas.
    if linea.startswith('"') and linea.endswith('"'):
        linea = linea[1:-1]
        linea = linea.replace('""', '"')
    try:
        fila = next(
            csv.reader(
                StringIO(linea),
                skipinitialspace=True
            )
        )
        filas.append(fila)

    except Exception as e:

        print("Fila con error:", linea)
        print("Error:", e)

columnas = filas[0]
datos = filas[1:]

df = pd.DataFrame(datos,columns=columnas)

print("========== DATASET ORIGINAL ==========")
print("Filas:", df.shape[0])
print("Columnas:", df.shape[1])

# 2. CONSERVAR LAS 1000 FILAS PRINCIPALES
df_principal = df.iloc[:1000].copy()

print("\n========== DATASET PRINCIPAL ==========")
print("Filas utilizadas:", len(df_principal))

# 3. LIMPIAR NOMBRES DE COLUMNAS
df_principal.columns = (df_principal.columns.astype(str).str.strip())

# 4. RECUPERAR ID DE VENTA
df_principal["id_venta"] = range(1,len(df_principal) + 1)

# 5. LIMPIAR PRODUCTO
df_principal["producto"] = (
    df_principal["producto"]
    .astype("string")
    .str.strip()
    .str.lower()
)

df_principal["producto"] = (
    df_principal["producto"]
    .replace(
        [  
             "",
            "nan",
            "none",
            "na"
        ],
        pd.NA
    )
)

# 6. NORMALIZAR PRODUCTOS
productos = {
    "arroz": "Arroz",
    "azucar": "Azucar",
    "azúcar": "Azucar",
    "aceite": "Aceite",
    "fideos": "Fideos",
    "atun": "Atún",
    "atún": "Atún",
    "leche": "Leche",
    "pan": "Pan",
    "huevos": "Huevos",
    "gaseosa": "Gaseosa",
    "detergente": "Detergente"
}
df_principal["producto"] = (df_principal["producto"].map(productos))

# 7. LIMPIAR CATEGORIA
df_principal["categoria"] = (
    df_principal["categoria"]
    .astype("string")
    .str.strip()
    .str.lower()
)

df_principal["categoria"] = (
    df_principal["categoria"]
    .replace(
        [
            "",
            "nan",
            "none",
            "na"
        ],
        pd.NA
    )
)

# 8. RELACIONAR PRODUCTO CON CATEGORIA
categoria_producto = {
    "Arroz": "abarrotes",
    "Azucar": "abarrotes",
    "Aceite": "abarrotes",
    "Fideos": "abarrotes",
    "Atún": "abarrotes",
    "Leche": "lacteos",
    "Pan": "panaderia",
    "Huevos": "huevos",
    "Gaseosa": "bebidas",
    "Detergente": "limpieza"
}

# Si conocemos el producto,
df_principal["categoria"] = (
    df_principal["producto"]
    .map(categoria_producto)
    .fillna(df_principal["categoria"])
)

# 9. RECUPERAR PRODUCTO USANDO CATEGORIA
producto_por_categoria = {
    "lacteos": "Leche",
    "panaderia": "Pan",
    "huevos": "Huevos",
    "bebidas": "Gaseosa",
    "limpieza": "Detergente"
}

df_principal["producto"] = (
    df_principal["producto"]
    .fillna(
        df_principal["categoria"]
        .map(producto_por_categoria)
    )
)

# 10. LIMPIAR METODO DE PAGO
df_principal["metodo_pago"] = (
    df_principal["metodo_pago"]
    .astype("string")
    .str.strip()
    .str.lower()
)

df_principal["metodo_pago"] = (
    df_principal["metodo_pago"]
    .replace(
        [
            "",
            "nan",
            "none",
            "na"
        ],
        pd.NA
    )
)

# 11. LIMPIAR CLIENTE FRECUENTE
df_principal["cliente_frecuente"] = (
    df_principal["cliente_frecuente"]
    .astype("string")
    .str.strip()
    .str.lower()
)

df_principal["cliente_frecuente"] = (
    df_principal["cliente_frecuente"]
    .replace(
        {
            "sí": "si",
            "si": "si",
            "no": "no",
            "": pd.NA,
            "nan": pd.NA,
            "none": pd.NA,
            "na": pd.NA
        }
    )
)

# 12. LIMPIAR OPINION DEL USUARIO
df_principal["opinion_usuario"] = (
    df_principal["opinion_usuario"]
    .astype("string")
    .str.lower()
    .str.replace(
        r"\s+",
        " ",
        regex=True
    )
    .str.strip()
)

df_principal["opinion_usuario"] = (
    df_principal["opinion_usuario"]
    .replace(
        [
            "",
            "nan",
            "none",
            "na"
        ],
        pd.NA
    )
)

# 13. CONVERTIR FECHA
df_principal["fecha"] = pd.to_datetime(
    df_principal["fecha"],
    errors="coerce"
)

# 14. CONVERTIR COLUMNAS NUMERICAS
columnas_numericas = [
    "cantidad",
    "precio_unitario",
    "descuento_pct",
    "satisfaccion",
    "total"
]

for columna in columnas_numericas:
    # Convertir coma decimal a punto.
    df_principal[columna] = (
        df_principal[columna]
        .astype("string")
        .str.strip()
        .str.replace(
            ",",
            ".",
            regex=False
        )
    )

    # Convertir a número.
    # Los valores imposibles pasan a NaN.
    df_principal[columna] = pd.to_numeric(df_principal[columna], errors="coerce")

# 15. CORREGIR SIGNOS NEGATIVOS
tolerancia = 0.10
for i in df_principal.index:
    cantidad = df_principal.at[
        i,
        "cantidad"
    ]
    precio = df_principal.at[
        i,
        "precio_unitario"
    ]
    descuento = df_principal.at[
        i,
        "descuento_pct"
    ]
    total = df_principal.at[
        i,
        "total"
    ]

    # CANTIDAD NEGATIVA
    if (
        pd.notna(cantidad)
        and cantidad < 0
        and pd.notna(precio)
        and precio > 0
        and pd.notna(descuento)
        and 0 <= descuento <= 100
        and pd.notna(total)
        and total >= 0
    ):
        cantidad_nueva = abs(cantidad)
        total_calculado = (
            cantidad_nueva
            * precio
            * (1 - descuento / 100)
        )
        if np.isclose(
            total_calculado,
            total,
            atol=tolerancia
        ):
            df_principal.at[
                i,
                "cantidad"
            ] = cantidad_nueva

    # PRECIO NEGATIVO
    precio = df_principal.at[
        i,
        "precio_unitario"
    ]

    if (
        pd.notna(precio)
        and precio < 0
        and pd.notna(cantidad)
        and cantidad > 0
        and pd.notna(descuento)
        and 0 <= descuento <= 100
        and pd.notna(total)
        and total >= 0
    ):
        precio_nuevo = abs(precio)
        total_calculado = (
            cantidad
            * precio_nuevo
            * (1 - descuento / 100)
        )
        if np.isclose(
            total_calculado,
            total,
            atol=tolerancia
        ):
            df_principal.at[
                i,
                "precio_unitario"
            ] = precio_nuevo

    # DESCUENTO NEGATIVO
    descuento = df_principal.at[
        i,
        "descuento_pct"
    ]
    if (
        pd.notna(descuento)
        and descuento < 0
        and pd.notna(cantidad)
        and cantidad > 0
        and pd.notna(precio)
        and precio > 0
        and pd.notna(total)
        and total >= 0
    ):
        descuento_nuevo = abs(
            descuento
        )
        if descuento_nuevo <= 100:
            total_calculado = (
                cantidad
                * precio
                * (
                    1
                    - descuento_nuevo / 100
                )
            )
            if np.isclose(
                total_calculado,
                total,
                atol=tolerancia
            ):
                df_principal.at[
                    i,
                    "descuento_pct"
                ] = descuento_nuevo

    # TOTAL NEGATIVO
    total = df_principal.at[
        i,
        "total"
    ]
    if (
        pd.notna(total)
        and total < 0
        and pd.notna(cantidad)
        and cantidad > 0
        and pd.notna(precio)
        and precio > 0
        and pd.notna(descuento)
        and 0 <= descuento <= 100
    ):
        total_nuevo = (
            cantidad
            * precio
            * (
                1
                - descuento / 100
            )
        )
        df_principal.at[
            i,
            "total"
        ] = round(
            total_nuevo,
            2
        )

# 16. VALIDAR VALORES NUMERICOS
# Cantidad debe ser mayor que 0.
df_principal.loc[
    df_principal["cantidad"] <= 0,
    "cantidad"
] = np.nan

# Precio debe ser mayor que 0.
df_principal.loc[
    df_principal["precio_unitario"] <= 0,
    "precio_unitario"
] = np.nan

# Descuento entre 0 y 100.
df_principal.loc[
    (df_principal["descuento_pct"] < 0)
    |
    (df_principal["descuento_pct"] > 100),
    "descuento_pct"
] = np.nan

# Satisfacción entre 1 y 5.
df_principal.loc[
    (df_principal["satisfaccion"] < 1)
    |
    (df_principal["satisfaccion"] > 5),
    "satisfaccion"
] = np.nan

# Total no puede ser negativo.
df_principal.loc[
    df_principal["total"] < 0,
    "total"
] = np.nan

# 17. RECUPERAR PRECIO UNITARIO
for i in df_principal.index:
    cantidad = df_principal.at[
        i,
        "cantidad"
    ]
    precio = df_principal.at[
        i,
        "precio_unitario"
    ]
    descuento = df_principal.at[
        i,
        "descuento_pct"
    ]
    total = df_principal.at[
        i,
        "total"
    ]
    if (
        pd.isna(precio)
        and pd.notna(cantidad)
        and pd.notna(descuento)
        and pd.notna(total)
    ):
        divisor = (
            cantidad
            * (1 - descuento / 100)
        )
        if divisor > 0:
            precio_calculado = (
                total / divisor
            )
            if precio_calculado > 0:

                df_principal.at[
                    i,
                    "precio_unitario"
                ] = round(
                    precio_calculado,
                    2
                )

# 18. RECUPERAR DESCUENTO
for i in df_principal.index:
    cantidad = df_principal.at[
        i,
        "cantidad"
    ]
    precio = df_principal.at[
        i,
        "precio_unitario"
    ]
    descuento = df_principal.at[
        i,
        "descuento_pct"
    ]
    total = df_principal.at[
        i,
        "total"
    ]

    if (
        pd.isna(descuento)
        and pd.notna(cantidad)
        and pd.notna(precio)
        and pd.notna(total)
    ):

        total_sin_descuento = (
            cantidad * precio
        )
        if total_sin_descuento > 0:
            descuento_calculado = (
                1
                - total / total_sin_descuento
            ) * 100

            if (
                0 <= descuento_calculado <= 100
            ):
                df_principal.at[
                    i,
                    "descuento_pct"
                ] = round(descuento_calculado,2)

# 19. RECUPERAR CANTIDAD
for i in df_principal.index:

    cantidad = df_principal.at[
        i,
        "cantidad"
    ]
    precio = df_principal.at[
        i,
        "precio_unitario"
    ]
    descuento = df_principal.at[
        i,
        "descuento_pct"
    ]
    total = df_principal.at[
        i,
        "total"
    ]

    if (
        pd.isna(cantidad)
        and pd.notna(precio)
        and pd.notna(descuento)
        and pd.notna(total)
    ):
        divisor = (
            precio
            * (1 - descuento / 100)
        )

        if divisor > 0:
            cantidad_calculada = (
                total / divisor
            )
            cantidad_entera = round(
                cantidad_calculada
            )

            # La cantidad debe ser un entero
            # y el cálculo debe estar cerca.
            if (
                cantidad_entera >= 1
                and np.isclose(
                    cantidad_calculada,
                    cantidad_entera,
                    atol=0.05
                )
            ):

                df_principal.at[
                    i,
                    "cantidad"
                ] = cantidad_entera

#20. RECUPERAR TOTAL
for i in df_principal.index:
    cantidad = df_principal.at[
        i,
        "cantidad"
    ]
    precio = df_principal.at[
        i,
        "precio_unitario"
    ]
    descuento = df_principal.at[
        i,
        "descuento_pct"
    ]
    total = df_principal.at[
        i,
        "total"
    ]
    if (
        pd.isna(total)
        and pd.notna(cantidad)
        and pd.notna(precio)
        and pd.notna(descuento)
    ):
        total_calculado = (cantidad * precio * (1- descuento / 100))
        df_principal.at[i,"total"] = round(total_calculado,2)

# 21. VALIDAR SATISFACCION
df_principal.loc[
    (df_principal["satisfaccion"] < 1)
    |
    (df_principal["satisfaccion"] > 5),
    "satisfaccion"
] = np.nan

# 22. FORMATEAR FECHA
df_principal["fecha"] = (df_principal["fecha"].dt.strftime("%Y-%m-%d"))

# 23. CREAR VARIABLES DE FECHA
fecha_temporal = pd.to_datetime(
    df_principal["fecha"],
    errors="coerce"
)

df_principal["año"] = (fecha_temporal.dt.year.astype("Int64"))
df_principal["mes"] = (fecha_temporal.dt.month.astype("Int64"))

# 24. NOMBRE DEL MES
meses = {
    1: "enero",
    2: "febrero",
    3: "marzo",
    4: "abril",
    5: "mayo",
    6: "junio",
    7: "julio",
    8: "agosto",
    9: "septiembre",
    10: "octubre",
    11: "noviembre",
    12: "diciembre"
}

df_principal["nombre_mes"] = (
    df_principal["mes"]
    .map(meses)
)

# 25. CREAR AÑO_MES
df_principal["año_mes"] = (
    fecha_temporal.dt.strftime("%Y-%m")
)

# 26. CONVERTIR VARIABLES DERIVADAS A TEXTO
columnas_derivadas = [
    "año",
    "mes",
    "nombre_mes",
    "año_mes"
]

for columna in columnas_derivadas:
    df_principal[columna] = (
        df_principal[columna]
        .astype("string")
    )

# 27. REEMPLAZAR FALTANTES SOLO EN TEXTO
columnas_texto = [
    "fecha",
    "producto",
    "categoria",
    "metodo_pago",
    "cliente_frecuente",
    "opinion_usuario",
    "año",
    "mes",
    "nombre_mes",
    "año_mes"
]

for columna in columnas_texto:
    df_principal[columna] = (
        df_principal[columna]
        .fillna("desconocido")
    )

# 28. ORDEN FINAL DE COLUMNAS
columnas_finales = [
    "id_venta",
    "fecha",
    "producto",
    "categoria",
    "cantidad",
    "precio_unitario",
    "descuento_pct",
    "metodo_pago",
    "cliente_frecuente",
    "satisfaccion",
    "total",
    "opinion_usuario",
    "año",
    "mes",
    "nombre_mes",
    "año_mes"
]

df_principal = (df_principal[columnas_finales])

# 29. GUARDAR DATASET LIMPIO
df_principal.to_csv("limpieza_datos.csv",index=False,encoding="utf-8-sig")

# 30. MOSTRAR RESULTADO FINAL
print("\n========== RESULTADO FINAL ==========")

print("Filas:",df_principal.shape[0])
print("Columnas:",df_principal.shape[1])

print("\nColumnas finales:")
print(list(df_principal.columns))

# 31. MOSTRAR VALORES FALTANTES
print("\n========== VALORES FALTANTES ==========")
print(df_principal.isna().sum())

# 32. MOSTRAR PRIMERAS 20 FILAS
print("\n========== PRIMERAS 20 FILAS ==========")
print(df_principal.head(20))

# 33. MENSAJE FINAL
print("\nArchivo generado:")
print("limpieza_datos.csv")