# Importamos las librerias necesarias
import numpy as np
import streamlit as st
import plotly.graph_objs as go

def grafica_continua(t, x_t, color, title):
    # Función para graficar una señal continua
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=t, y=x_t, mode="lines", name=title, line=dict(color=color))
    )

    fig.update_layout(
        xaxis_title="Tiempo",
        yaxis_title="Amplitud",
        showlegend=True,
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    st.plotly_chart(fig, use_container_width=True)

def grafica_frec(t, x_t, color, title):
    # Función para graficar una señal continua
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=t, y=x_t, mode="lines", name=title, line=dict(color=color))
    )

    fig.update_layout(
        xaxis_title="Frecuencia (Hz)",
        yaxis_title="Magnitud",
        showlegend=True,
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    st.plotly_chart(fig, use_container_width=True)

# Definimos las funciones necesarias
def calc_fft(signal):
    w = np.linspace(-len(signal)/2, len(signal)/2, len(signal))
    fft = np.fft.fft(signal)
    fft_centered = np.abs(np.fft.fftshift(fft))
    fft_normalized = fft_centered/np.max(fft_centered)
    return w, fft_normalized

def filtro_pasabajas(signal, w_0, fs):
    w = 2*np.pi* np.fft.fftfreq(len(signal), d=1/fs)
    x_w = np.fft.fft(signal)
    filter = np.abs(w) <= w_0
    x_w_filtered = x_w * filter
    filtered_signal = np.fft.ifft(x_w_filtered)
    return np.real(filtered_signal)

# Configuración inicial de la aplicación en Streamlit
st.set_page_config(layout="wide")
st.title("Modulación de Amplitud")

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

# Sidebar for parameters
st.sidebar.header("Parametros de la señal")

# Adjustable parameters
A1 = st.sidebar.number_input("Amplitud de la señal 1", value=1.0, key="a1")
A2 = st.sidebar.number_input("Amplitud de la señal 2", value=1.0, key="a2")
fs = st.sidebar.slider("Frecuencia de muestreo (Hz)", 2500, 3000, 2500)
f_1 = st.sidebar.slider("Frecuencia de modulación de la señal 1 (Hz)", 1, 20, 5)
f_2 = st.sidebar.slider("Frecuencia de modulación de la señal 2 (Hz)", 1, 20, 7)
fo = st.sidebar.slider("Frecuencia de la portadora (Hz)", 12000, 20000, 12000)
w_cutoff = st.sidebar.slider("Frecuencia de corte del filtro (Hz)", 1, 50, 10)


t = np.arange(0, 1, 1/fs)

x1_t = A1*np.cos(2 * np.pi * f_1 * t)
x2_t = A2*np.cos(2 * np.pi * f_2 * t)

w0 = 2 * np.pi * fo
p1_t = np.cos(w0 * t)
p2_t = np.sin(w0 * t)

# FFT para modular señales
w_1, x1_w_normalized = calc_fft(x1_t)
w_2, x2_w_normalized = calc_fft(x2_t)

w_p1, p1_w_normalized = calc_fft(p1_t)
w_p2, p2_w_normalized = calc_fft(p2_t)

# Modulación
x1_mod = x1_t * p1_t
x2_mod = x2_t * p2_t
y_t = x1_mod + x2_mod

# FFT para modular señales
w_1_mod, x1_mod_w_normalized = calc_fft(x1_mod)
w_2_mod, x2_mod_w_normalized = calc_fft(x2_mod)
w_total, y_w_normalized = calc_fft(y_t)

# Demodulación
y1_dem = y_t * p1_t
y2_dem = y_t * p2_t

# FFT para demodular señales
w_1_dem, y1_dem_w_normalized = calc_fft(y1_dem)
w_2_dem, y2_dem_w_normalized = calc_fft(y2_dem)

col1, col2 = st.columns(2)
# Visualización de las señales
st.header("Señales de entrada")

with col1:
    grafica_continua(t, x1_t, "blue", 'Señal 1 - Dominio del tiempo')
    grafica_continua(t, x2_t, "red", 'Señal 2 - Dominio del tiempo')

with col2:
    grafica_frec(w_1, x1_w_normalized, "blue", 'Señal 1 - Dominio de la frecuencia')
    grafica_frec(w_2, x2_w_normalized, "red", 'Señal 2 - Dominio de la frecuencia')

st.header("Señales moduladas")
col1, col2 = st.columns(2)

with col1:
    grafica_continua(t, x1_mod, "blue", 'Señal 1 Modulada - Dominio del tiempo')
    grafica_continua(t, x2_mod, "red", 'Señal 2 Modulada - Dominio del tiempo')

with col2:
    grafica_frec(w_1_mod, x1_mod_w_normalized, "blue", 'Señal 1 Modulada - Dominio de la frecuencia')
    grafica_frec(w_2_mod, x2_mod_w_normalized, "red", 'Señal 2 Modulada - Dominio de la frecuencia')

st.header("Señales demoduladas")
col1, col2 = st.columns(2)

with col1:
    grafica_continua(t, y1_dem, "blue", 'Señal 1 Demodulada - Dominio del tiempo')
    grafica_continua(t, y2_dem, "red", 'Señal 2 Demodulada - Dominio del tiempo')

with col2:
    grafica_frec(w_1_dem, y1_dem_w_normalized, "blue", 'Señal 1 Demodulada - Dominio de la frecuencia')
    grafica_frec(w_2_dem, y2_dem_w_normalized, "red", 'Señal 2 Demodulada - Dominio de la frecuencia')

st.header("Filtro pasabajas")
col1, col2 = st.columns(2)
y1_recovered = filtro_pasabajas(y1_dem, 2*np.pi*w_cutoff, fs)
y2_recovered = filtro_pasabajas(y2_dem, 2*np.pi*w_cutoff, fs)

# FFT para señales recuperadas
w_1_rec, y1_recovered_w_normalized = calc_fft(y1_recovered)
w_2_rec, y2_recovered_w_normalized = calc_fft(y2_recovered)

col1, col2 = st.columns(2)
with col1:
    grafica_continua(t, y1_recovered, "blue", 'Señal 1 Recuperada - Dominio del tiempo')

with col2:
    grafica_continua(t, y2_recovered, "red", 'Señal 2 Recuperada - Dominio del tiempo')