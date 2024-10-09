# Importamos las librerias necesarias
import numpy as np
import plotly.graph_objs as go
import streamlit as st
import time

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

def graficar(t, x_t):
    plot_placeholder = st.empty()

    # Crear la figura inicial
    trace = go.Scatter(x=t, y=x_t, mode='lines', name='Señal Invertida')
    layout = go.Layout(xaxis=dict(range=[-21, 21]), yaxis=dict(range=[-3, 3]), title='Señal Invertida en Movimiento')
    fig = go.Figure(data=[trace], layout=layout)

    # Actualizar la señal para moverla de izquierda a derecha en el eje X
    for shift in np.linspace(-20, 20, 20):
        # Actualizar los valores de X para moverlos
        new_x = t + shift
        fig.data[0].x = new_x
        
        # Renderizar la figura actualizada en el mismo lienzo
        plot_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Agregar un pequeño retardo para ver la animación
        time.sleep(0.3)


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
                t_xinv, x_inv = invertir(x, y)
                grafica_continua(t_xinv, x_inv, "green", "x(t) invertida")
            elif inv == "h(t)":
                t_hinv, h_inv = invertir(h, z)
                grafica_continua(t_hinv, h_inv, "green", "h(t) invertida")
