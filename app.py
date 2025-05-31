import streamlit as st
import pandas as pd
import pickle

# Cargar modelo
@st.cache_resource
def cargar_modelo():
    with open("best_model (17).pkl", "rb") as file:
        return pickle.load(file)

modelo = cargar_modelo()

# Cargar DataFrame de ejemplo
df = pd.read_excel("dataframe_exportado.xlsx")

# Obtener columnas y calcular valores promedio
columnas = df.drop("Estado Aprendiz", axis=1).columns
valores_default = df.drop("Estado Aprendiz", axis=1).mean()

st.title("Predicción del Estado del Aprendiz")

# Entrada de usuario
edad = st.slider("Edad", 18, 100, 25)
cantidad_quejas = st.selectbox("Cantidad de quejas", list(range(0, 11)))
estrato = st.selectbox("Estrato socioeconómico", [1, 2, 3, 4, 5, 6])

# Crear nueva muestra para predicción
nueva_muestra = valores_default.copy()
nueva_muestra["Edad"] = edad
nueva_muestra["Cantidad de quejas"] = cantidad_quejas
nueva_muestra["Estrato"] = estrato

entrada_modelo = pd.DataFrame([nueva_muestra])

# Realizar predicción
prediccion = modelo.predict(entrada_modelo)[0]

# Mostrar resultado
st.subheader("Resultado de la predicción:")
st.write(f"📊 Estado del aprendiz predicho: **{prediccion}**")
