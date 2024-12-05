# Importamos las librerías necesarias
import streamlit as st

# Configuración inicial de la aplicación en Streamlit
st.set_page_config(layout="wide")
st.title("Interfaz gráfica de series y transformada de Fourier")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.page_link("streamlit_app.py", label="Inicio", use_container_width=True)

with col2:
    st.page_link("pages/primero.py", label="Ejemplo de Series", use_container_width=True)

with col3:
    st.page_link("pages/segundo.py", label="Modulación de Señales", use_container_width=True)

with col4:
    st.page_link("pages/tercero.py", label="Modulación de Amplitud", use_container_width=True)
with col5:
    st.page_link("pages/cuarto.py", label="DSB-LC", use_container_width=True)

st.sidebar.markdown("Creado por: Oliver Ardila, Juan Bermejo y Daniel Henriquez")

st.markdown("#### Agradecimientos ####")
col1, col2 = st.columns(2)
with col1:
    st.image("aura.jpg", width=400)
with col2:
    st.image("lena.png", width=400)

st.markdown(
    "Damos gracias a Cristiano Ronaldo y a Lena Gray por el desarrollo de esta interfaz gráfica."
)
