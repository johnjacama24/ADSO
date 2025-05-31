import streamlit as st
import pandas as pd
import pickle

# ----------------------------
# Cargar el modelo .pkl
# ----------------------------
@st.cache(allow_output_mutation=True)
def cargar_modelo():
    with open("best_model.pkl", "rb") as file:
        return pickle.load(file)

modelo = cargar_modelo()

# ----------------------------
# Cargar el dataframe original
# ----------------------------
df = pd.read_excel("dataframe.xlsx", engine="openpyxl")

# ----------------------------
# Configuración de la app
# ----------------------------
st.title("Predicción del Estado del Aprendiz")
st.write("Complete la información para predecir el estado del aprendiz.")

# Entradas del usuario
edad = st.slider("Edad", 18, 100, 25)
cantidad_quejas = st.selectbox("Cantidad de quejas", list(range(0, 11)))
estrato = st.selectbox("Estrato socioeconómico", [1, 2, 3, 4, 5, 6])

# ----------------------------
# Preparar entrada para el modelo
# ----------------------------

# Obtener nombres y orden de columnas como las espera el modelo
columnas_modelo = df.drop("Estado Aprendiz", axis=1).columns

# Crear muestra con valores promedio
valores_default = df.drop("Estado Aprendiz", axis=1).mean()
nueva_muestra = valores_default.copy()

# Reemplazar los valores ingresados
nueva_muestra["Edad"] = edad
nueva_muestra["Cantidad de quejas"] = cantidad_quejas
nueva_muestra["Estrato"] = estrato

# Convertir en DataFrame con columnas en el orden original
entrada_modelo = pd.DataFrame([nueva_muestra])[columnas_modelo]

# ----------------------------
# Realizar la predicción
# ----------------------------
try:
    prediccion = modelo.predict(entrada_modelo)[0]
    st.subheader("Resultado de la predicción:")
    st.success(f"📊 Estado del aprendiz predicho: **{prediccion}**")
except Exception as e:
    st.error("❌ Error al hacer la predicción:")
    st.code(str(e))
