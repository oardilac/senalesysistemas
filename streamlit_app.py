# Importamos las librerias necesarias
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import sympy as sp
from scipy.interpolate import interp1d

# Vectores de tiempo
Delta = 0.01  # Tiempo 1 Continua
t1 = np.arange(-2, -1, Delta)
t2 = np.arange(-1, 1, Delta)
t3 = np.arange(1, 2 + Delta, Delta)
t1_T = np.concatenate((t1, t2, t3))  # Tiempo total


t1_2 = np.arange(-3, -2, Delta)  # Tiempo 2 Continua
t2_2 = np.arange(-2, -1, Delta)
t3_2 = np.arange(-1, 0, Delta)
t4_2 = np.arange(0, 2, Delta)
t5_2 = np.arange(2, 3, Delta)
t6_2 = np.arange(3, 3 + Delta, Delta)
t2_T = np.concatenate((t1_2, t2_2, t3_2, t4_2, t5_2, t6_2))  # Tiempo total

n1 = np.arange(-5, 16 + 1)  # Secuencia discreta 1


n2_1 = np.arange(-10, -6 + 1)  # Secuencia discreta 2
n2_2 = np.arange(-5, 0 + 1)
n2_3 = np.arange(1, 5 + 1)
n2_4 = np.arange(6, 10 + 1)
n2 = np.concatenate((n2_1, n2_2, n2_3, n2_4))  # Tiempo total

# Declaración de Funciones y secuencias
x1_1 = 2 * t1 + 4
x1_2 = 2 * np.ones(len(t2))
x1_3 = -2 * t3 + 4
x_t1 = np.concatenate((x1_1, x1_2, x1_3))  # Funcion 1 Continua


x2_1 = t1_2 + 3
x2_2 = 2 * np.ones(len(t2_2))
x2_3 = t3_2 + 3
x2_4 = -t4_2 + 3
x2_5 = np.ones(len(t5_2))
x_t2 = np.concatenate((x2_1, x2_2, x2_3, x2_4, x2_5, [0]))  # Funcion 2 continua


x_n = [
    0,
    0,
    0,
    0,
    0,
    -3,
    0,
    5,
    4,
    -2,
    -4,
    -1,
    2,
    5,
    7,
    4,
    -2,
    0,
    0,
    0,
    0,
    0,
]  # Secuencia Discreta

x_n2_1 = np.zeros(len(n2_1))

x_n2_2 = np.zeros(len(n2_2))
for j in range(len(n2_2)):
    x_n2_2[j] = (2 / 3) ** (n2_2[j])

x_n2_3 = np.zeros(len(n2_3))
for j in range(len(n2_3)):
    x_n2_3[j] = (8 / 5) ** (n2_3[j])

x_n2_4 = np.zeros(len(n2_4))

x_n2 = np.concatenate((x_n2_1, x_n2_2, x_n2_3, x_n2_4))  # Secuencia Discreta 2


# Metodo 1 Tiempo Continuo
def metodo1(t, f, a, t0):
    x = sp.Symbol("x")
    t1 = t - t0  # Desplazamiento temporal
    tesc = t1 / a  # Escalamiento temporal

    # Define un rango común para el eje x
    x_min = min(t1.min(), tesc.min(), t.min())
    x_max = max(t1.max(), tesc.max(), t.max())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=t, y=f, mode="lines", line=dict(color="blue"), name=f"Señal ({x})")
    )
    fig.update_layout(
        title=f"Señal Original: ({x})",
        xaxis_title="Tiempo",
        yaxis_title="Amplitud",
        legend_title="Transformación",
        grid=dict(rows=1, columns=1),
        showlegend=False,
        xaxis_range=[x_min, x_max],
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    # Gráfico 1: Señal desplazada en el tiempo
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=t1,
            y=f,
            mode="lines",
            name=f"Señal Desplazada ({x + t0})",
            line=dict(color="green"),
        )
    )
    fig1.update_layout(
        title=f"Señal Desplazada ({x + t0})",
        xaxis_title="Tiempo",
        yaxis_title="Amplitud",
        legend_title="Transformación",
        grid=dict(rows=1, columns=1),
        showlegend=False,
        xaxis_range=[x_min, x_max],
    )

    fig1.update_xaxes(showgrid=True)
    fig1.update_yaxes(showgrid=True)

    # Gráfico 2: Señal escalada y desplazada en el tiempo
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=tesc,
            y=f,
            mode="lines",
            name=f"Señal Desplazada y Escalada ({a*x + t0})",
            line=dict(color="red"),
        )
    )
    fig2.update_layout(
        title=f"Señal Desplazada y Escalada: ({a*x + t0})",
        xaxis_title="Tiempo",
        yaxis_title="Amplitud",
        legend_title="Transformación",
        grid=dict(rows=1, columns=1),
        showlegend=False,
        xaxis_range=[x_min, x_max],
    )

    fig2.update_xaxes(showgrid=True)
    fig2.update_yaxes(showgrid=True)

    # Mostrar las tres graficas
    with st.container():
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)


def metodo2(t, f, a, t0):  # Metodo 2 transformación
    x = sp.Symbol("x")
    tesc = t / a
    t1 = tesc - (t0 / a)
    # Define un rango común para el eje x
    x_min = min(t1.min(), tesc.min(), t.min())
    x_max = max(t1.max(), tesc.max(), t.max())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=t, y=f, mode="lines", line=dict(color="blue"), name=f"Señal ({x})")
    )
    fig.update_layout(
        title=f"Señal Original: ({x})",
        xaxis_title="Tiempo",
        yaxis_title="Amplitud",
        legend_title="Transformación",
        grid=dict(rows=1, columns=1),
        showlegend=False,
        xaxis_range=[x_min, x_max],
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    # Gráfico 1: Señal escalada en el tiempo
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=t1,
            y=f,
            mode="lines",
            name=f"Señal Escalada ({a*x})",
            line=dict(color="green"),
        )
    )
    fig1.update_layout(
        title=f"Señal Escalada ({a*x})",
        xaxis_title="Tiempo",
        yaxis_title="Amplitud",
        legend_title="Transformación",
        grid=dict(rows=1, columns=1),
        showlegend=False,
        xaxis_range=[x_min, x_max],
    )

    fig1.update_xaxes(showgrid=True)
    fig1.update_yaxes(showgrid=True)

    # Gráfico 2: Señal desplazada y escalada en el tiempo
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=tesc,
            y=f,
            mode="lines",
            name=f"Señal Escalada y Desplazada ({a*(x+t0/a)})",
            line=dict(color="red"),
        )
    )
    fig2.update_layout(
        title=f"Señal Escalada y Desplazada: ({a*(x+t0/a)})",
        xaxis_title="Tiempo",
        yaxis_title="Amplitud",
        legend_title="Transformación",
        grid=dict(rows=1, columns=1),
        showlegend=False,
        xaxis_range=[x_min, x_max],
    )

    fig2.update_xaxes(showgrid=True)
    fig2.update_yaxes(showgrid=True)

    # Mostrar las tres graficas
    with st.container():
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)


def stem(n, f, title, color):
    fig = go.Figure()

    # Añadir las líneas de los "stems"
    for x_val, y_val in zip(n, f):
        fig.add_trace(
            go.Scatter(
                x=[x_val, x_val],  # misma coordenada x
                y=[0, y_val],  # desde el eje hasta el valor en y
                mode="lines",  # solo dibuja la línea
                line=dict(color=color, dash="dash"),
                showlegend=False,
            )
        )

    # Añadir los marcadores
    fig.add_trace(
        go.Scatter(
            x=n,
            y=f,
            mode="markers",
            marker=dict(color=color, size=10),
            name=title,
        )
    )

    # Actualizar layout del gráfico
    fig.update_layout(
        title=title,
        xaxis=dict(tickmode="array", tickvals=n),
        xaxis_title="Tiempo",
        yaxis_title="Amplitud",
        showlegend=False,
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    # Mostrar gráfico
    st.plotly_chart(fig, use_container_width=True)


def metodo1D(n, f, M, n0):  # Metodo 1 Discreta
    stem(n, f, "Señal Original", "yellow")
    if abs(M) < 1:
        n1 = n - n0

        # Gráfico 1 - Secuencia Desplazada
        stem(n1, f, "Secuencia Desplazada", "green")
        Z=int(1/abs(M))
        L_n=len(f)
        nI=np.arange(n1[0]*Z,(n1[-1]*Z)+1)
        L_nI=len(nI)
        x_nI0=np.arange(1,L_nI+1)
        x_nIEsc=np.arange(1,L_nI+1)

        for k in range(L_nI-1):   #Escalon y Cero
            if k % Z ==0:
                r=int(k*abs(M))
                x_nI0[k]=f[r]
                x_nIEsc[k]=f[r]
            else:
                x_nI0[k]=0
                x_nIEsc[k]=x_nIEsc[k-1]
        x_nI0[L_nI-1]=f[L_n-1]
        x_nIEsc[L_nI-1]=f[L_n-1]

        x_nM = np.arange(1, L_nI+1, dtype=float)
        k=0
        for s in range (L_n-1):   #Lineal
            x_nM[k] = f[s]
            for j in range (1,Z):
                dif = f[s+1] - f[s]
                A = j*abs(M)*dif + f[s]
                x_nM[k+1] = A
                k=k+1
            k=k+1
        x_nM[L_nI-1] = f[L_n-1]
        if M>0:
            # Gráfico 2 - Cero
            stem(nI,x_nI0,"Cero","red")

            # Gráfico 3 - Escalón
            stem(nI,x_nIEsc,"Escalón","blue")

            # Gráfico 4 - Lineal
            stem(nI,x_nM,"Lineal","orange")
        else:
            for i in range(len(nI)):
                nI[i]=nI[i]*-1
            
            # Gráfico 2 - Cero
            stem(nI,x_nI0,"Cero","red")

            # Gráfico 3 - Escalón
            stem(nI,x_nIEsc,"Escalón","blue")

            # Gráfico 4 - Lineal
            stem(nI,x_nM,"Lineal","orange")
    else:
        n1 = n - n0
        new = []
        tnd = []
        for i in range(len(n1)):
            if n1[i] % abs(M) == 0:
                new.append(n1[i] / abs(M))
                tnd.append(f[i])
        stem(n1, f, "Secuencia Desplazada", "green")
        if M > 0:
            # Gráfico 2 - Secuencia Transformada
            stem(new, tnd, "Secuencia Desplazada y Escalonada", "red")
        else:
            for i in range(len(new)):
                new[i] = new[i] * -1
            # Gráfico 2 - Secuencia Transformada
            stem(new, tnd, "Secuencia Desplazada y Escalonada", "red")


def metodo2D(n, f, M, n0):
    stem(n, f, "Señal Original", "yellow")
    if abs(M) < 1:
        Z=int(1/abs(M))
        L_n=len(f)
        nI=np.arange(n[0]*Z,(n[-1]*Z)+1)
        L_nI=len(nI)
        x_nI0=np.arange(1,L_nI+1)
        x_nIEsc=np.arange(1,L_nI+1)
        for k in range(L_nI-1):   #Escalon y Cero
            if k % Z ==0:
                r=int(k*abs(M))
                x_nI0[k]=f[r]
                x_nIEsc[k]=f[r]
            else:
                x_nI0[k]=0
                x_nIEsc[k]=x_nIEsc[k-1]
        x_nI0[L_nI-1]=f[L_n-1]
        x_nIEsc[L_nI-1]=f[L_n-1]

        x_nM = np.arange(1, L_nI+1, dtype=float)
        k=0
        for s in range (L_n-1):   #Lineal
            x_nM[k] = f[s]
            for j in range (1,Z):
                dif = f[s+1] - f[s]
                A = j*abs(M)*dif + f[s]
                x_nM[k+1] = A
                k=k+1
            k=k+1
        x_nM[L_nI-1] = f[L_n-1]
        
        # Gráfico 2 - Cero
        stem(nI,x_nI0,"Cero","red")

        # Gráfico 3 - Escalón
        stem(nI,x_nIEsc,"Escalón","blue")

        # Gráfico 4 - Lineal
        stem(nI,x_nM,"Lineal","orange")

        if M > 0:  # Interpolación positiva
            fc = n0 / abs(M)
            nI = nI - fc

            # Gráfico 5 - Cero (desplazado)
            stem(nI, x_nI0, "Cero Desplazado", "pink")

            # Gráfico 6 - Escalón (desplazado)
            stem(nI, x_nIEsc, "Escalón Desplazado", "purple")

            # Gráfico 7 - Lineal (desplazado)
            stem(nI, x_nM, "Lineal Desplazado", "brown")
        else:
            fc = n0 / abs(M)
            nI = nI - fc
            for i in range(len(nI)):
                nI[i] = nI[i] * -1
            # Gráfico 4 - Cero (invertido)
            stem(nI, x_nI0, "Cero Desplazado", "pink")

            # Gráfico 5 - Escalón (invertido)
            stem(nI, x_nIEsc, "Escalón Desplazado", "purple")

            # Gráfico 6 - Lineal (invertido)
            stem(nI, x_nM, "Lineal Desplazado", "brown")

    else:
        new=[]
        tnd=[]
        for i in range(len(n)):
            if n[i] % abs(M)==0:
                new.append(n[i]/abs(M))
                tnd.append(f[i])
        stem(new, tnd, "Secuencia Escalonada", "green")
        n1=n-n0
        new2=[]
        tnd2=[]
        for i in range(len(n1)):
            if n1[i] % abs(M)==0:
                new2.append(n1[i]/abs(M))
                tnd2.append(f[i])
        if M > 0:
            # Gráfico 2 - Transformación positiva
            stem(new2, tnd2, "Secuencia Escalonada y Desplazada", "blue")
        else:
            for i in range(len(new2)):
                new2[i]=new2[i]*-1
            # Gráfico 2 - Transformación negativa
            stem(new2, tnd2, "Secuencia Escalonada y Desplazada", "blue")


def suma(t, f):  # Suma para tiempo continuo
    t1 = t - (1 / 4)
    tesc = t1 / (-1 / 3)

    t2 = t - (-1 / 3)
    tesc2 = t2 / (1 / 2)

    interp_f1 = interp1d(tesc, f, kind="linear", fill_value=0, bounds_error=False)
    interp_f2 = interp1d(tesc2, f, kind="linear", fill_value=0, bounds_error=False)
    min_bound = min(tesc.min(), tesc2.min())
    max_bound = max(tesc.max(), tesc2.max())
    comun_tdf = np.linspace(min_bound, max_bound, 15000)
    x_t1_interp = interp_f1(comun_tdf)
    x_t2interp = interp_f2(comun_tdf)
    x_tsuma = x_t1_interp + x_t2interp

    col1, col2 = st.columns(2)
    # Primera columna: 1/4 - t/3
    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=tesc, y=f, mode="lines", name="1/4 - t/3"))

        fig1.update_layout(
            xaxis_title="Tiempo",
            yaxis_title="Amplitud",
            showlegend=True,
        )

        fig1.update_xaxes(showgrid=True)
        fig1.update_yaxes(showgrid=True)

        # Mostrar gráfico en la primera columna
        st.plotly_chart(fig1, use_container_width=True)

    # Segunda columna: t/2 - 1/3
    with col2:
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=tesc2,
                y=f,
                mode="lines",
                name="t/2 - 1/3",
                line=dict(color="green"),
            )
        )
        fig2.update_layout(
            xaxis_title="Tiempo",
            yaxis_title="Amplitud",
            showlegend=True,
        )

        fig2.update_xaxes(showgrid=True)
        fig2.update_yaxes(showgrid=True)

        # Mostrar gráfico en la segunda columna
        st.plotly_chart(fig2, use_container_width=True)

    # Segundo gráfico (Suma de las funciones interpoladas)
    fig_sum = go.Figure()
    fig_sum.add_trace(
        go.Scatter(
            x=comun_tdf,
            y=x_tsuma,
            mode="lines",
            name="Suma de las Señales Continuas",
            line=dict(color="red"),
        )
    )

    fig_sum.update_layout(
        title="Suma de las Señales Continuas",
        xaxis_title="Tiempo",
        yaxis_title="Amplitud",
        showlegend=False,
    )

    fig_sum.update_xaxes(showgrid=True)
    fig_sum.update_yaxes(showgrid=True)

    st.plotly_chart(fig_sum, use_container_width=True)


def sumad(n, f):
    M1 = 1 / 4
    n1 = n - (-3)
    Z1 = int(1 / (1 / 4))
    L_n = len(f)
    nI = np.arange(n1[0] * Z1, (n1[-1] * Z1) + 1)
    L_nI = len(nI)
    x_nM = np.arange(1, L_nI + 1, dtype=float)
    k = 0
    for i in range(L_n - 1):  # Lineal 1
        x_nM[k] = f[i]
        for j in range(1, Z1):
            dif = f[i + 1] - f[i]
            A = j * M1 * dif + f[i]
            x_nM[k + 1] = A
            k = k + 1
        k = k + 1
    x_nM[L_nI - 1] = f[L_n - 1]

    M2 = -1 / 3
    n2 = n - (4)
    Z2 = int(1 / abs(M2))
    nI2 = np.arange(n2[0] * Z2, (n2[-1] * Z2) + 1)
    L_nI2 = len(nI2)
    x_nM2 = np.arange(1, L_nI2 + 1, dtype=float)

    k = 0
    for i in range(L_n - 1):  # Lineal 2
        x_nM2[k] = f[i]
        for j in range(1, Z2):
            dif = f[i + 1] - f[i]
            A = j * abs(M2) * dif + f[i]
            x_nM2[k + 1] = A
            k = k + 1
        k = k + 1
    x_nM2[L_nI2 - 1] = f[L_n - 1]
    for i in range(len(nI2)):
        nI2[i] = nI2[i] * -1

    col1, col2 = st.columns(2)
    with col1:
        # Gráfico 1: Secuencia 1
        stem(nI, x_nM, "x[(n/4)-3]", "green")

    with col2:
        # Gráfico 2: Secuencia 2
        stem(nI2, x_nM2, "x[4-(n/3)]", "red")

    min_n = min(nI.min(), nI2.min())
    max_n = max(nI.max(), nI2.max())
    comun_n = np.arange(min_n, max_n + 1)  # Tiempo Comun

    interp_n1 = interp1d(nI, x_nM, bounds_error=False, fill_value=0)
    interp_n2 = interp1d(nI2, x_nM2, bounds_error=False, fill_value=0)

    x_nMC = interp_n1(comun_n)
    x_nM2C = interp_n2(comun_n)

    x_nsuma = x_nMC + x_nM2C

    # Crear el tercer gráfico de stem (suma)
    stem(comun_n, x_nsuma, "Suma de Secuencias", "blue")


st.set_page_config(layout="wide")
st.title("Interfaz gráfica de procesamiento de señales")

st.sidebar.title("Menu de operaciones")
operation = st.sidebar.selectbox(
    "Tipo de Señal", ["Seleccionar...", "Continua", "Discreta"]
)

if operation == "Seleccionar...":
    st.header("El bicho")
    st.image("aura.jpg", caption="Julio Voltio", use_column_width=True)

elif operation == "Continua":
    signal = st.sidebar.radio("Señal", ["1", "2"])
    if signal == "1":
        x = t1_T
        y = x_t1
    else:
        x = t2_T
        y = x_t2

    sum = st.sidebar.radio(
        "Suma",
        ["No", "Si"],
        key="sum_continua",
    )
    if sum == "No":
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
        )
        t0 = st.sidebar.select_slider(
            "t0",
            [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6],
            value=1,
        )

        if method == "Desplazamiento/Escalamiento":
            metodo1(x, y, a, t0)
        else:
            metodo2(x, y, a, t0)

    else:
        suma(x, y)

elif operation == "Discreta":
    signal = st.sidebar.radio("Señal", ["1", "2"])
    if signal == "1":
        x = n1
        y = x_n
    else:
        x = n2
        y = x_n2

    sum = st.sidebar.radio(
        "Suma",
        ["No", "Si"],
        key="sum_discreto",
    )
    if sum == "No":
        method = st.sidebar.radio(
            "Metodo",
            ["Desplazamiento/Escalamiento", "Escalamiento/Desplazamiento"],
            key="method_discreta",
        )

        M = st.sidebar.select_slider(
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
        n0 = st.sidebar.select_slider(
            "n0",
            [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6],
            value=1,
        )

        if method == "Desplazamiento/Escalamiento":
            metodo1D(x, y, M, n0)
        else:
            metodo2D(x, y, M, n0)

    else:
        sumad(x, y)
