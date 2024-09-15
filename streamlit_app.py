# Importamos las librerias necesarias
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import sympy as sp

#Vectores de tiempo
Delta = 0.01    # Tiempo 1 Continua
t1 = np.arange(-2, -1, Delta)
t2 = np.arange(-1, 1, Delta)
t3 = np.arange(1, 2+Delta, Delta)
t1_T=np.concatenate((t1, t2, t3))  #Tiempo total


t1_2=np.arange(-3, -2, Delta)   #Tiempo 2 Continua
t2_2=np.arange(-2, -1, Delta)
t3_2=np.arange(-1, 0, Delta)
t4_2=np.arange(0, 2, Delta)
t5_2=np.arange(2, 3, Delta)
t6_2=np.arange(3, 3+Delta, Delta)
t2_T=np.concatenate((t1_2, t2_2,t3_2,t4_2,t5_2,t6_2))  #Tiempo total

n1=np.arange(-5,16+1)  #Secuencia discreta 1


n2_1=np.arange(-10,-6+1)    #Secuencia discreta 2
n2_2=np.arange(-5,0+1)
n2_3=np.arange(1,5+1)
n2_4=np.arange(6,10+1)
n2=np.concatenate((n2_1, n2_2, n2_3,n2_4))  #Tiempo total

#Declaración de Funciones y secuencias
x1_1=2*t1+4
x1_2=2*np.ones(len(t2))
x1_3=-2*t3+4
x_t1 = np.concatenate((x1_1, x1_2,x1_3 ))   #Funcion 1 Continua


x2_1=t1_2+3
x2_2=2*np.ones(len(t2_2))
x2_3=t3_2+3
x2_4=-t4_2+3
x2_5=np.ones(len(t5_2))
x_t2 = np.concatenate((x2_1, x2_2,x2_3,x2_4,x2_5,[0]))         #Funcion 2 continua


x_n=[0,0,0,0,0,-3,0,5,4,-2,-4,-1,2,5,7,4,-2,0,0,0,0,0]  #Secuencia Discreta

x_n2_1=np.zeros(len(n2_1))

x_n2_2=np.zeros(len(n2_2))
for j in range(len(n2_2)):
   x_n2_2[j]=(2/3)**(n2_2[j])

x_n2_3=np.zeros(len(n2_3))
for j in range(len(n2_3)):
   x_n2_3[j]=(8/5)**(n2_3[j])

x_n2_4=np.zeros(len(n2_4))

x_n2=np.concatenate((x_n2_1, x_n2_2,x_n2_3,x_n2_4 ))  #Secuencia Discreta 2


#Metodo 1 Tiempo Continuo
def metodo1(t, f, a, t0):
    x = sp.Symbol("x")
    t1 = t - t0  # Desplazamiento temporal
    tesc = t1 / a  # Escalamiento temporal
    
    # Define un rango común para el eje x
    x_min = min(t1.min(), tesc.min(), t.min())
    x_max = max(t1.max(), tesc.max(), t.max())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=f,
            mode="lines",
            line=dict(color="blue"),
            name=f'Señal ({x})'
        )
    )
    fig.update_layout(
        title=f"Señal Original: ({x})",
        xaxis_title="Tiempo",
        yaxis_title="Amplitud",
        legend_title="Transformación",
        grid=dict(rows=1, columns=1),
        showlegend=False,
        xaxis_range=[x_min, x_max]
    )

    # Gráfico 1: Señal desplazada en el tiempo
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=t1, y=f, mode='lines', name=f'Señal Desplazada ({x + t0})', line=dict(color='green')))
    fig1.update_layout(
        title=f"Señal Desplazada ({x + t0})",
        xaxis_title="Tiempo",
        yaxis_title="Amplitud",
        legend_title="Transformación",
        grid=dict(rows=1, columns=1),
        showlegend=False,
        xaxis_range=[x_min, x_max]
    )

    # Gráfico 2: Señal escalada en el tiempo
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=tesc, y=f, mode='lines', name=f'Señal Escalada (({a*x + t0})', line=dict(color='red')))
    fig2.update_layout(
        title=f"Señal Escalada: (({a*x + t0})",
        xaxis_title="Tiempo",
        yaxis_title="Amplitud",
        legend_title="Transformación",
        grid=dict(rows=1, columns=1),
        showlegend=False,
        xaxis_range=[x_min, x_max]
    )

    # Mostrar las tres graficas
    with st.container():
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)

st.set_page_config(layout="wide")
st.title("Interfaz grafica de procesamiento de señales")

col1, col2 = st.columns([1, 3])
st.sidebar.title("Menu de operaciones")
operation = st.sidebar.selectbox(
    "Operaciones", ["Seleccionar...", "Continua", "Discreta", "Suma"]
)
signal = st.sidebar.radio("Señal", ["1", "2"])

if operation == "Seleccionar...":
    st.header("El bicho")
    st.image("aura.jpg", caption="Julio Voltio", use_column_width=True)

elif operation == "Continua":
    if signal == "1":
        x = t1_T
        y=x_t1
    else:
        x=t2_T
        y=x_t2
    
    method = st.sidebar.radio(
        "Metodo",
        ["Desplazamiento/Escalamiento", "Escalamiento/Desplazamiento"],
        key="method_continua",
    )
    
    a = st.sidebar.select_slider(
        "a",
        np.round(
            [
                -5,
                -4,
                -3,
                -2,
                -1 / 2,
                -1 / 3,
                -1 / 4,
                -1 / 5,
                1 / 5,
                1 / 4,
                1 / 3,
                1 / 2,
                2,
                3,
                4,
                5,
            ],
            2,
        ),
        value=2,
        key="a",
        label_visibility="collapsed",
    )
    t0 = st.sidebar.select_slider(
        "t0",
        [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6],
        value=1,
    )

    if method == "Desplazamiento/Escalamiento":
        metodo1(x, y, a, t0)

elif operation == "Discreta":
    if signal == "1":
        fig = go.Figure()

        for x_val, y_val in zip(n1, x_n):
            fig.add_trace(
                go.Scatter(
                    x=[x_val, x_val],  # misma coordenada x
                    y=[0, y_val],  # desde el eje hasta el valor en y
                    mode="lines",  # solo dibuja la línea
                    line=dict(color="green", dash="dash"),
                    showlegend=False
                )
            )

        # Añade los marcadores
        fig.add_trace(
            go.Scatter(
                x=n1,
                y=x_n,
                mode="markers",
                marker=dict(color="green", size=10),
                name="Señal Discreta 1",
            )
        )

        fig.update_layout(
            title="Señal Discreta 1",
            xaxis=dict(tickmode="array", tickvals=n2),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = go.Figure()
        # Añade las líneas verticales desde el eje x hasta los puntos
        for x_val, y_val in zip(n2, x_n2):
            fig.add_trace(
                go.Scatter(
                    x=[x_val, x_val],  # misma coordenada x
                    y=[0, y_val],  # desde el eje hasta el valor en y
                    mode="lines",  # solo dibuja la línea
                    line=dict(color="purple", dash="dash"),
                    showlegend=False
                )
            )

        # Añade los marcadores
        fig.add_trace(
            go.Scatter(
                x=n2,
                y=x_n2,
                mode="markers",
                marker=dict(color="purple", size=10),
                name="Señal Discreta 2",
            )
        )

        fig.update_layout(
            title="Señal Discreta 2",
            xaxis=dict(tickmode="array", tickvals=n2),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    method = st.sidebar.radio(
        "Metodo",
        ["Desplazamiento/Escalamiento", "Escalamiento/Desplazamiento"],
        key="method_discreta",
    )

    a = st.sidebar.select_slider(
        "M",
        np.round(
            [
                -4,
                -3,
                -2,
                -1 / 2,
                -1 / 3,
                -1 / 4,
                -1 / 5,
                1 / 5,
                1 / 4,
                1 / 3,
                1 / 2,
                2,
                3,
                4,
            ],
            2,
        ),
        value=2,
    )
    t0 = st.sidebar.select_slider(
        "n0",
        [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6],
        value=1,
    )

else:
    pass