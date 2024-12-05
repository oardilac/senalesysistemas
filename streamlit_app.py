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

st.markdown("Las herramientas más esenciales en el análisis de señales y sistemas incluyen la serie y la transformada de Fourier. Su capacidad para descomponer señales en componentes sinusoidales ha sido clave en su aplicación en múltiples ámbitos. Estos conceptos son fundamentales en áreas como las telecomunicaciones, donde se utilizan para procesos como la modulación y demodulación en términos de amplitud o frecuencia.")
st.markdown("En el tratamiento de señales, tanto la serie como la transformada de Fourier tienen un rol crucial en el diseño de filtros, permitiendo seleccionar o eliminar ciertas frecuencias en función de los requerimientos específicos de un sistema. Asimismo, en el campo del audio y sonido, estas herramientas resultan indispensables para tareas como la grabación, compresión y reproducción, además de optimizar el diseño de los circuitos asociados.")
st.markdown("Por otro lado, el algoritmo Fast Fourier Transform (FFT) permite una implementación eficiente de la transformada discreta en distintas áreas. La FFT no solo facilita la ejecución de análisis que de otro modo serían más complejos, sino que también mejora significativamente la eficiencia en el cálculo de la transformada, especialmente en el manejo de señales de gran tamaño.")

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
