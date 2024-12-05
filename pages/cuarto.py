import streamlit as st
import numpy as np
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

# Create a row with two columns to place buttons side by side
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

st.title("Signal Processing and Modulation Visualization")

# Sidebar for input parameters
st.sidebar.header("Signal Parameters")

# Input for the three signals
with st.sidebar.expander("Señal 1"):
    f1 = st.number_input("Frecuencia de señal 1 (Hz)", value=100.0, key="f1")
    A1 = st.number_input("Amplitud de la señal 1", value=1.0, key="a1")

with st.sidebar.expander("Señal 2"):
    f2 = st.number_input("Frecuencia de señal 2 (Hz)", value=200.0, key="f2")
    A2 = st.number_input("Amplitud de la señal 2", value=0.5, key="a2")

with st.sidebar.expander("Señal 3"):
    f3 = st.number_input("Frecuencia de señal 3 (Hz)", value=300.0, key="f3")
    A3 = st.number_input("Amplitud de la señal 3", value=0.3, key="a3")

# Sampling parameters
fs = 100000  # Sampling frequency (Hz)
t = np.linspace(0, 0.015, int(fs * 0.015), endpoint=False)

# Generate signals
y1 = A1 * np.sin(2 * np.pi * f1 * t)
y2 = A2 * np.sin(2 * np.pi * f2 * t)
y3 = A3 * np.sin(2 * np.pi * f3 * t)
y_t = y1 + y2 + y3

# Plot individual signals
st.header("Señales individuales y compuestas")

grafica_continua(t, y1, "blue", "Señal 1")
grafica_continua(t, y2, "red", "Señal 2")
grafica_continua(t, y3, "green", "Señal 3")
grafica_continua(t, y_t, "purple", "Señal compuesta")

# Carrier signal and modulation
st.header("Análisis de la modulación de la señal")

# Carrier signal parameters
A_p = 1  # Carrier amplitude
f0 = 3000  # Carrier frequency (Hz)
p_t = A_p * np.cos(2 * np.pi * f0 * t)  # Carrier signal
y_mod = y_t * p_t  # Amplitude modulation (DSB-SC)

# Frequency domain analysis
w = np.linspace(-len(y_mod)/2, len(y_mod)/2, len(y_mod))

# Calculate FFTs
y_w = np.fft.fftshift(np.abs(np.fft.fft(y_t)))
y_w_norm = y_w / np.max(y_w)

p_t_w = np.fft.fftshift(np.abs(np.fft.fft(p_t)))
p_t_w_norm = p_t_w / np.max(p_t_w)

y_mod_w = np.fft.fftshift(np.abs(np.fft.fft(y_mod)))
y_mod_w_norm = y_mod_w / np.max(y_mod_w)

col1, col2 = st.columns(2)

with col1:
    grafica_continua(t, y_t, "blue", "Señal Moduladora")
    grafica_continua(t, p_t, "red", "Señal Portadora")
    grafica_continua(t, y_mod, "green", "Señal Modulada")

with col2:
    grafica_frec(w, y_w_norm, "blue", "Espectro Señal Moduladora")
    grafica_frec(w, p_t_w_norm, "red", "Espectro Señal Portadora")
    grafica_frec(w, y_mod_w_norm, "green", "Espectro Señal Modulada")

# Primera sección de índices de modulación
st.header("DSB-LC Analisis Modulación")

# Índices de modulación fijos para la primera sección
mu_fixed = [1.2, 1.0, 0.7]

a_m = np.max(np.abs(y_t))

col1, col2 = st.columns(2)

for i, m in enumerate(mu_fixed):
    A_p = a_m / m
    y_mod = (A_p + y_t) * np.cos(2*np.pi*f0*t)
    
    # FFT de la señal modulada
    y_mod_w = np.fft.fftshift(np.abs(np.fft.fft(y_mod)))
    y_mod_w_norm = y_mod_w / np.max(y_mod_w)
    
    # Señal modulada
    with col1:
        grafica_continua(t, y_mod, "green", f"Señal Modulada (μ = {m})")
    
    with col2:
        grafica_frec(w, y_mod_w_norm, "red", f"Espectro Señal Modulada (μ = {m})")

# Segunda sección de índices de modulación con señales rectificadas
st.header("DSB-LC Modulación Analisis con rectificación")

col1, col2 = st.columns(2)
for i, m in enumerate(mu_fixed):
    A_p = a_m / m
    y_mod = (A_p + y_t) * np.cos(2*np.pi*f0*t)
    y_mod_rect = np.abs(y_mod)

    with col1:
        grafica_continua(t, y_mod, "green", f"Señal Modulada (μ = {m})")
    
    with col2:
        grafica_continua(t, y_mod_rect, "orange", f"Señal Rectificada (μ = {m})")
    