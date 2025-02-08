# Importamos las librerías necesarias
import numpy as np
import plotly.graph_objs as go
import streamlit as st
import time
from scipy.interpolate import interp1d

Delta = 0.01  # Delta para definir el paso de tiempo

def invertir(t, x_t):
    # Invertir una señal continua
    return -t[::-1], x_t[::-1]


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


def convolucion_continua(t, x_t, h, h_t, key_prefix=""):
    # Realizar la convolución de dos señales continuas
    y_conv = np.convolve(x_t, h_t) * Delta  # Convolución numérica usando numpy
    t_conv = np.arange(t[0] + h[0], t[0] + h[0] + len(y_conv) * Delta, Delta)

    # Invertir la señal h(t) para la convolución
    h, h_t = invertir(h, h_t)

    interp_func = interp1d(t_conv, y_conv, bounds_error=False, fill_value=0)
    x_min = min(t.min() - 5, h.min() - 1)
    x_max = max(t.max() + 5, h.max() + 1)

    # Configuración de las columnas para mostrar las gráficas en paralelo
    col1, col2 = st.columns(2)

    plot_placeholder_1 = col1.empty()
    plot_placeholder_2 = col2.empty()

    # Gráfica de las señales fija y móvil
    trace_fija = go.Scatter(x=t, y=x_t, mode="lines", name="Señal Fija")
    trace_movil = go.Scatter(x=h, y=h_t, mode="lines", name="Señal en Movimiento")

    layout_señales = go.Layout(
        xaxis=dict(showgrid=True, range=[x_min, x_max]),
        yaxis=dict(showgrid=True),
        title="Señales: Fija y en Movimiento",
    )

    fig_señales = go.Figure(data=[trace_fija, trace_movil], layout=layout_señales)

    # Configuración para el rango de movimiento de la señal móvil
    shift_min = t[0] - 7 - h[0]
    shift_max = t[-1] + 7 - h[-1]

    x_full = np.arange(shift_min, shift_max, Delta)
    y_full = interp_func(x_full)

    # Gráfica de la convolución
    trace_convolucion = go.Scatter(x=x_full, y=y_full, mode="lines", name="Convolución")

    layout_convolucion = go.Layout(
        xaxis=dict(showgrid=True, autorange=True),
        yaxis=dict(showgrid=True),
        title="Convolución",
    )

    fig_convolucion = go.Figure(data=[trace_convolucion], layout=layout_convolucion)

    plot_placeholder_2.plotly_chart(
        fig_convolucion, use_container_width=True, key=key_prefix + "_convolution_chart"
    )

    # Animación de la convolución, moviendo la señal móvil
    for j in range(len(x_full)):
        new_h = h + x_full[j]
        fig_señales.data[1].x = new_h

        fig_convolucion.data[0].y = y_full[: j + 1]

        plot_placeholder_1.plotly_chart(
            fig_señales, use_container_width=True, key=key_prefix + f"_signal_chart_{j}"
        )
        plot_placeholder_2.plotly_chart(
            fig_convolucion, use_container_width=True, key=key_prefix + f"_convolution_chart_{j}"
        )
        time.sleep(0.01)  # Pausa para la animación

# Configuración inicial de la aplicación en Streamlit
st.set_page_config(layout="wide")
st.title("Interfaz gráfica de convolución de señales")

st.sidebar.markdown("Creado por: Oliver Ardila y Jesus Carmona")

bw = st.sidebar.slider(
    "Seleccione el ancho de banda del sistema",
    min_value=0.1,
    max_value=2.0,
    value=1.0,    # valor por defecto
    step=0.1
)

pe = 1 / bw  # Periodo de la señal

ta = [-pe/2]
ta2 = np.arange(-pe/2, pe/2, Delta)
ta3 = [pe/2]
ta_T = np.concatenate((ta, ta2, ta3))  # Tiempo total para señal a

# Definición de las señales continuas
a = [0]
xa2 = np.ones(len(ta2))
xa3 = [0]
x_ta = np.concatenate((a, xa2, xa3))  # Función continua para señal a

# Lista de valores para τ
taus = [1, 0.8, 0.5, 0.3, 0.1]

td = [0]
td2 = np.arange(0, 5, Delta)
td_T = np.concatenate((td, td2))  # Tiempo total para señal d

xd = [1]
xd2 = np.exp(-td2)  # Exponencial positiva para valores negativos de tiempo
x_td = np.concatenate((xd, xd2))  # Función continua para señal d

# Graficamos las señales individuales
col1, col2 = st.columns(2)
with col1:
    grafica_continua(ta_T, x_ta, "blue", "x(t)")
with col2:
    grafica_continua(td_T, x_td, "red", f"h(t) con τ = 1")

st.markdown("#### Proceso de convolución ####")

# Iteramos sobre cada valor de τ para generar y graficar la convolución
for tau in taus:
    st.markdown(f"### Convolución con τ = {tau} ###")

    # Definición de la señal d, que será la exponencial
    td2 = np.arange(-0.01, 5, Delta)
    # Aquí se define la exponencial con la forma (1/τ)*e^(-t/τ)
    xd2 = (1/tau) * np.exp(-td2/tau)

    # Llamamos a la función de convolución, pasando un prefijo único basado en τ
    convolucion_continua(ta_T, x_ta, td2, xd2, key_prefix=f"tau_{tau}")