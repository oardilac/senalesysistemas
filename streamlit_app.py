# Importamos las librerias necesarias
import numpy as np
import plotly.graph_objs as go
import streamlit as st
import time
from scipy.interpolate import interp1d

Delta = 0.01    # Delta

ta=[0]
ta2 = np.arange(0, 3, Delta)
ta3 = np.arange(3, 5, Delta)
ta4 = [5]
ta_T=np.concatenate((ta, ta2, ta3,ta4))  #Tiempo total a

tb = [-1]
tb2 = np.arange(-1, 1, Delta)
tb3 = [1]
tb_T=np.concatenate((tb, tb2, tb3))  #Tiempo total b

tc = [-1]
tc2 = np.arange(-1, 1, Delta)
tc3 = np.arange(1, 3, Delta)
tc4 = np.arange(3, 5, Delta)
tc5 = [5]
tc_T=np.concatenate((tc, tc2, tc3,tc4,tc5))  #Tiempo total c

td = [-3]
td2 = np.arange(-3, 0, Delta)
td3 = np.arange(0, 3, Delta)
td4 = [3]
td_T=np.concatenate((td, td2, td3,td4))  #Tiempo total d

xa=[0]
xa2=2*np.ones(len(ta2))
xa3=-2*np.ones(len(ta3))
xa4=[0]
x_ta = np.concatenate((xa, xa2,xa3,xa4 ))   #Funcion a Continua


xb=[0]
xb2=-tb2*1
xb3=[0]
x_tb = np.concatenate((xb, xb2,xb3 ))   #Funcion b Continua


xc=[0]
xc2=2*np.ones(len(tc2))
xc3=-2*tc3+4
xc4=-2*np.ones(len(tc4))
xc5=[0]
x_tc = np.concatenate((xc, xc2,xc3,xc4,xc5 ))   #Funcion c Continua

xd=[0]
xd2=np.exp(td2)
xd3=np.exp(-td3)
xd4=[0]
x_td = np.concatenate((xd, xd2,xd3,xd4 ))   #Funcion d Continua

def invertir(t, x_t):
    return -t[::-1], x_t[::-1]

def grafica_continua(t, x_t, color, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=x_t, mode="lines", name=title, line=dict(color=color)))

    fig.update_layout(
        xaxis_title="Tiempo",
        yaxis_title="Amplitud",
        showlegend=True,
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    # Mostrar gráfico en la primera columna
    st.plotly_chart(fig, use_container_width=True)


def graficar(t, x_t, h, h_t):
    # Convolución usando numpy (sin hacer en cada paso)
    y_conv = np.convolve(x_t, h_t) * Delta
    # Ajustar el tiempo para la convolución
    t_conv = np.arange(t[0] + h[0], t[0] + h[0] + len(y_conv) * Delta, Delta)

    # Invertir la señal h(t)
    h, h_t = invertir(h, h_t)
    
    interp_func = interp1d(t_conv, y_conv, bounds_error=False, fill_value=0)
    # Definir los límites de las gráficas
    x_min = min(t.min()-5, h.min()-1)
    x_max = max(t.max()+5, h.max()+1)

    # Dividir en dos columnas
    col1, col2 = st.columns(2)

    # Contenedor de la animación (col1 para señales)
    plot_placeholder_1 = col1.empty()
    # Contenedor para la convolución (col2 para convolución)
    plot_placeholder_2 = col2.empty()

    # Crear la traza para la señal fija (x_t)
    trace_fija = go.Scatter(x=t, y=x_t, mode='lines', name='Señal Fija')

    # Crear la traza para la señal en movimiento (h_t)
    trace_movil = go.Scatter(x=h, y=h_t, mode='lines', name='Señal en Movimiento')

    # Configuración del layout para señales
    layout_señales = go.Layout(
        xaxis=dict(showgrid=True, range=[x_min, x_max]),
        yaxis=dict(showgrid=True),
        title='Señales: Fija y en Movimiento'
    )

    # Crear la figura con ambas señales en movimiento
    fig_señales = go.Figure(data=[trace_fija, trace_movil], layout=layout_señales)

    # Rango de movimiento de la señal móvil
    shift_min = t[0] - 7 - h[0]
    shift_max = t[-1] + 7 - h[-1]

    x_full = np.arange(shift_min, shift_max, Delta)
    y_full = interp_func(x_full)

    # Crear la traza para la convolución (vacía por ahora)
    trace_convolucion = go.Scatter(x=x_full, y=y_full, mode='lines', name='Convolución')

    layout_convolucion = go.Layout(
        xaxis=dict(showgrid=True, autorange=True),
        yaxis=dict(showgrid=True),
        title='Convolución'
    )

    # Crear la figura con la convolución
    fig_convolucion = go.Figure(data=[trace_convolucion], layout=layout_convolucion)

    # Renderizar la figura actualizada en col2 (convolución)
    plot_placeholder_2.plotly_chart(fig_convolucion, use_container_width=True, key="convolution_chart")

    # Animar la señal en movimiento (h_t)
    for j in range(len(x_full)):
        # Actualizar los valores de X para mover la señal en movimiento
        new_h = h + x_full[j]
        fig_señales.data[1].x = new_h  # Actualizamos solo la señal en movimiento

        # Renderizar la figura actualizada en col1 (señales en movimiento)
        fig_convolucion.data[0].y = y_full[:j+1]
        
        plot_placeholder_1.plotly_chart(fig_señales, use_container_width=True, key=f"signal_chart_{j}")

        # Renderizar la figura actualizada en col2 (convolución)
        plot_placeholder_2.plotly_chart(fig_convolucion, use_container_width=True, key=f"convolution_chart_{j}")
        # Agregar un pequeño retardo para la animación final
        time.sleep(0.01)

st.set_page_config(layout="wide")
st.title("Interfaz gráfica de convolución de señales")

st.sidebar.title("Menu de operaciones")
operation = st.sidebar.selectbox(
    "Tipo de Señal", ["Menú Inicial...", "Continua", "Discreta"]
)

if operation == "Menú Inicial...":
    st.sidebar.markdown("Creado por: Oliver Ardila, Juan Bermejo y Daniel Henriquez")

    st.markdown("#### Agradecimientos ####")
    col1, col2 = st.columns(2)
    with col1:
        st.image("aura.jpg", width=400)
    with col2:
        st.image("lena.png", width=400)

    st.markdown("Damos gracias a Cristiano Ronaldo y a Lena Gray por el desarrollo de esta interfaz gráfica.")
elif operation == "Continua":
    x_t = st.sidebar.selectbox("Señal x(t)", ["Seleccione", "A", "B", "C", "D"])
    h_t = st.sidebar.selectbox("Señal h(t)", ["Seleccione", "A", "B", "C", "D"])

    if x_t == "Seleccione" or h_t == "Seleccione":
        st.error("Seleccione las señales a graficar")
    else:
        if x_t == "A":
            x = ta_T
            y = x_ta
        elif x_t == "B":
            x = tb_T
            y = x_tb
        elif x_t == "C":
            x = tc_T
            y = x_tc
        elif x_t == "D":
            x = td_T
            y = x_td

        if h_t == "A":
            h = ta_T
            z = x_ta
        elif h_t == "B":
            h = tb_T
            z = x_tb
        elif h_t == "C":
            h = tc_T
            z = x_tc
        elif h_t == "D":
            h = td_T
            z = x_td

        inv = st.sidebar.selectbox("Cual señal desea invertir", ["Seleccione", "x(t)", "h(t)"])
        col1, col2 = st.columns(2)
        with col1:
            grafica_continua(x, y, "blue", "x(t)")
        with col2:
            grafica_continua(h, z, "red", "h(t)")

        if inv == "Seleccione":
            st.error("Seleccione la señal a invertir")
        else:
            if inv == "x(t)":
                t_inv, x_inv = invertir(x, y)
                grafica_continua(t_inv, x_inv, "green", "x(t) invertida")
                st.markdown("### Proceso de convolución ###")
                graficar(h, z, x, y)
            elif inv == "h(t)":
                t_inv, x_inv = invertir(h, z)
                grafica_continua(t_inv, x_inv, "green", "h(t) invertida")
                st.markdown("### Proceso de convolución ###")
                graficar(x, y, h, z)