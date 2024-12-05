import numpy as np
from scipy.io import wavfile
import streamlit as st
import plotly.graph_objs as go
import io

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

def create_audio_file(signal, samplerate):
    """Convert signal to WAV file bytes"""
    # Normalize the signal to 16-bit range
    normalized = np.int16(signal * 32767)
    # Create a BytesIO buffer
    buffer = io.BytesIO()
    # Write the WAV file to the buffer
    wavfile.write(buffer, samplerate, normalized)
    # Get the buffer value
    buffer.seek(0)
    return buffer

# Configuración inicial de la aplicación en Streamlit
st.set_page_config(layout="wide")
st.title("Modulación de Señales")

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

# File upload
uploaded_file = st.file_uploader("Subir un archivo de audio WAV (.wav)", type=['wav'])

if uploaded_file is not None:
    # Read audio file
    try:
        # Convert uploaded file to a format that wavfile.read can handle
        audio_bytes = uploaded_file.read()
        audio_stream = io.BytesIO(audio_bytes)
        fs, data = wavfile.read(audio_stream)
        
        # Show basic audio information
        length = data.shape[0] / fs
        st.write(f"Frecuencia de muestreo: {fs} Hz")
        st.write(f"Duración: {length:.2f} segundos")

        # Create columns for audio players
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Audio original")
            # Reset the uploaded file position
            uploaded_file.seek(0)
            st.audio(uploaded_file)
        
        # Create time vector
        t = np.linspace(0, length, data.shape[0])
        
        # Process audio (mono or stereo)
        if len(data.shape) == 2 and data.shape[1] == 2:
            st.write("Audio Estereo detectado")
            x_t = np.mean(data, axis=1)/np.max(np.mean(data, axis=1))
        else:
            st.write("Mono audio detectado")
            x_t = data/np.max(data)

        # Carrier parameters in sidebar
        st.sidebar.header("Parametros de la modulación")
        A = st.sidebar.slider("Amplitud de la portadora", 0.1, 2.0, 1.0)

        # Carrier frequency selector
        f = st.sidebar.slider("Frecuencia de la portadora (Hz)", 
                                min_value=90000,
                                max_value=200000,
                                value=100000)
        
        # Filter cutoff frequency
        w_corte = st.sidebar.slider("Frecuencia de corte (Hz)",
                                    min_value=1000,
                                    max_value=20000,
                                    value=5000)

        n = len(x_t)
        tiempo = n / fs       # Duración del audio
        Delta_t = 1 / fs      # Tiempo de muestreo
        t = np.arange(n) * Delta_t

        # Cálculo del espectro
        X_w = np.fft.fft(x_t)            # Transformada de Fourier de x(t)
        X_w_cent = np.fft.fftshift(X_w)  # Centramos el espectro
        Delta_f = 1 / (n * Delta_t)
        f = np.arange(-n / 2, n / 2) * Delta_f
        #magnitud = np.abs(X_w_cent) / np.max
        magnitud = np.abs(X_w_cent) / n

        # Filtro pasa bajas
        fpb = np.abs(f) <= w_corte  # Frecuencia de corte, en este caso 1000 Hz

        X_w_fil = X_w_cent * fpb

        X_w_filt_corrida = np.fft.ifftshift(X_w_fil)    # Corrimiento del espectro
        x_t_filt = np.fft.ifft(X_w_filt_corrida)       # Transformada inversa

        w=2*np.pi*f
        pt=A*np.cos(w*t)
        fsp=fs*60
        T=1/fsp
        n=len(x_t_filt )
        t1=np.arange(n)*T
        pt2=A*np.cos(w*t1)   #Señal portadora

        # Cálculo del espectro
        yt=x_t_filt*pt
        Y_w = np.fft.fft(yt)            # Transformada de Fourier de x(t)
        Y_w_cent = np.fft.fftshift(Y_w)  # Centramos el espectro
        Delta_f = 1 / (n * Delta_t)
        w_p = np.linspace(-len(pt)/2,len(pt)/2,len(pt))
        magnitud = np.abs(Y_w_cent) / n

        xtdem=yt*pt

        # Cálculo del espectro
        X_wdem = np.fft.fft(xtdem)            # Transformada de Fourier de x(t)
        X_w_demcent = np.fft.fftshift(X_wdem)  # Centramos el espectro
        Delta_f = 1 / (n * Delta_t)
        magnitud = np.abs(X_w_demcent) / n
        f = np.arange(-len(xtdem)/2, len(xtdem)/2) * Delta_f

        X_w_fil2 = X_w_demcent * fpb
        XWrecup = np.fft.ifftshift(X_w_fil2)    # Corrimiento del espectro
        x_t_filt2= np.fft.ifft(XWrecup)       # Transformada inversa

        with col2:
            st.subheader("Audio filtrado")
            recovered_audio = create_audio_file(x_t_filt, fs)
            st.audio(recovered_audio)

        grafica_continua(t, x_t, "blue", "Señal original")
        grafica_frec(f, magnitud, "red", "Espectro de la señal original")
        grafica_frec(f, fpb, "green", "Filtro pasa bajas")
        grafica_frec(f, np.abs(X_w_fil) / np.max(np.abs(X_w_fil)), "purple", "Espectro de la señal filtrada")
        grafica_continua(t, pt2, "orange", "Señal portadora")
        grafica_frec(w_p, np.abs(Y_w_cent) / np.max(np.abs(X_w_fil)), "purple", "Espectro de la señal modulada")
        grafica_frec(f, np.abs(X_w_demcent) / np.max(np.abs(X_w_demcent)), "yellow", "Espectro de la señal demodulada")
        grafica_frec(f, np.abs(X_w_fil) / np.max(np.abs(X_w_fil)), "grey", "Espectro de la señal filtrada")
    except Exception as e:
        st.error(f"Error procesando archivo: {str(e)}")
else:
    st.info("Por favor suba un archivo de audio WAV (.wav)")