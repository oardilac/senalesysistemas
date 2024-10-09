# Importamos las librerías necesarias
import numpy as np
import plotly.graph_objs as go
import streamlit as st
import time
from scipy.interpolate import interp1d

Delta = 0.01  # Delta para definir el paso de tiempo

# Definición de los intervalos de tiempo para las señales a, b, c, d
ta = [0]
ta2 = np.arange(0, 3, Delta)
ta3 = np.arange(3, 5, Delta)
ta4 = [5]
ta_T = np.concatenate((ta, ta2, ta3, ta4))  # Tiempo total para señal a

tb = [-1]
tb2 = np.arange(-1, 1, Delta)
tb3 = [1]
tb_T = np.concatenate((tb, tb2, tb3))  # Tiempo total para señal b

tc = [-1]
tc2 = np.arange(-1, 1, Delta)
tc3 = np.arange(1, 3, Delta)
tc4 = np.arange(3, 5, Delta)
tc5 = [5]
tc_T = np.concatenate((tc, tc2, tc3, tc4, tc5))  # Tiempo total para señal c

td = [-3]
td2 = np.arange(-3, 0, Delta)
td3 = np.arange(0, 3, Delta)
td4 = [3]
td_T = np.concatenate((td, td2, td3, td4))  # Tiempo total para señal d

# Definición de las señales continuas
a = [0]
xa2 = 2 * np.ones(len(ta2))
xa3 = -2 * np.ones(len(ta3))
xa4 = [0]
x_ta = np.concatenate((a, xa2, xa3, xa4))  # Función continua para señal a

xb = [0]
xb2 = -tb2
xb3 = [0]
x_tb = np.concatenate((xb, xb2, xb3))  # Función continua para señal b

xc = [0]
xc2 = 2 * np.ones(len(tc2))
xc3 = -2 * tc3 + 4
xc4 = -2 * np.ones(len(tc4))
xc5 = [0]
x_tc = np.concatenate((xc, xc2, xc3, xc4, xc5))  # Función continua para señal c

xd = [0]
xd2 = np.exp(td2)  # Exponencial positiva para valores negativos de tiempo
d3 = np.exp(-td3)  # Exponencial decreciente para valores positivos de tiempo
d4 = [0]
x_td = np.concatenate((xd, xd2, d3, d4))  # Función continua para señal d

# Definición de señales discretas
na = np.arange(-5, 6)
xn_a = np.zeros(len(na))
for i in range(len(na)):
    xn_a[i] = 6 - abs(na[i])  # Definición de una señal triangular discreta

ha = np.arange(-5, 6)
hn_a = np.ones(len(ha))  # Definición de una señal constante discreta

nb = np.arange(-2, 9)
xn_b = np.ones(len(nb))  # Definición de una señal constante discreta

hb = np.arange(-1, 10)
hn_b = np.zeros(len(hb))
for i in range(len(hb)):
    hn_b[i] = (9 / 11) ** hb[i]  # Decaimiento exponencial discreto


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


def convolucion_continua(t, x_t, h, h_t):
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
        fig_convolucion, use_container_width=True, key="convolution_chart"
    )

    # Animación de la convolución, moviendo la señal móvil
    for j in range(len(x_full)):
        new_h = h + x_full[j]
        fig_señales.data[1].x = new_h

        fig_convolucion.data[0].y = y_full[: j + 1]

        plot_placeholder_1.plotly_chart(
            fig_señales, use_container_width=True, key=f"signal_chart_{j}"
        )
        plot_placeholder_2.plotly_chart(
            fig_convolucion, use_container_width=True, key=f"convolution_chart_{j}"
        )
        time.sleep(0.01)  # Pausa para la animación


def stem(n, f, title, color):
    # Función para graficar una señal discreta
    line_x = []
    line_y = []
    for x_val, y_val in zip(n, f):
        line_x.extend([x_val, x_val, None])  # Añadir None para separar las líneas
        line_y.extend([0, y_val, None])

    # Crear la traza para las líneas verticales
    lines = go.Scatter(
        x=line_x, y=line_y, mode="lines", line=dict(color=color), showlegend=False
    )

    # Crear la traza para los marcadores
    markers = go.Scatter(
        x=n, y=f, mode="markers", marker=dict(color=color, size=10), name=title
    )

    fig = go.Figure(data=[lines, markers])

    fig.update_layout(
        title=title,
        xaxis=dict(tickmode="array", tickvals=n),
        xaxis_title="Tiempo",
        yaxis_title="Amplitud",
        showlegend=True,
        template="plotly_white",
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig


def invertir_discreta(x, y):
    # Invertir una señal discreta
    return [-1 * i for i in x[::-1]], y[::-1]


def convolucion_discreta(x, h, x_n, h_n):
    # Realizar la convolución de dos señales discretas
    y_n = np.convolve(x_n, h_n)
    n_conv = np.arange(x[0] + h[0], x[0] + h[0] + len(y_n))
    h, h_n = invertir_discreta(h, h_n)
    interp_func = interp1d(n_conv, y_n, bounds_error=False, fill_value=0)

    n_min = min(min(x) - 5, min(h) - 1)
    n_max = max(max(x) + 5, max(h) + 1)

    # Gráfica de la señal fija
    fig_senales_fija = stem(x, x_n, "Señal Fija", "red")
    fig_senales_fija.update_layout(
        xaxis=dict(showgrid=True, range=[n_min, n_max]),
        yaxis=dict(showgrid=True),
        title="Señales: Fija y en Movimiento",
    )

    shift_min = x[0] - 20 - h[0]
    shift_max = x[-1] + 20 - h[-1]

    x_full = np.arange(shift_min, shift_max)
    y_full = interp_func(x_full)

    fig_convolucion = stem(x_full, y_full, "Convolución", "green")

    col1, col2 = st.columns(2)
    plot_placeholder_1 = col1.empty()
    plot_placeholder_2 = col2.empty()

    plot_placeholder_1.plotly_chart(
        fig_senales_fija, use_container_width=True, key="signals_chart_initial"
    )

    fig_senales_movil = go.Figure()

    fig_convolucion = go.Figure()

    plot_placeholder_2.plotly_chart(
        fig_convolucion, use_container_width=True, key="convolution_chart_initial"
    )

    # Animación de la convolución discreta
    for shift in range(len(y_full)):
        # Desplazar la señal móvil
        new_h = h + x_full[shift]
        fig_senales_movil = stem(new_h, h_n, "Señal en Movimiento", "blue")

        # Combinar la señal fija y la señal móvil desplazada
        fig_combined = go.Figure(data=fig_senales_fija.data + fig_senales_movil.data)

        fig_combined.update_layout(
            title="Señales: Fija y en Movimiento",
            xaxis=dict(showgrid=True, range=[n_min, n_max]),
            yaxis=dict(showgrid=True),
            template="plotly_white",
        )

        plot_placeholder_1.plotly_chart(
            fig_combined, use_container_width=True, key=f"signals_chart_{shift}"
        )

        # Actualizar la convolución parcial
        y_partial = y_full[: shift + 1]
        n_partial = x_full[: shift + 1]

        fig_convolucion = stem(n_partial, y_partial, "Convolucion", "green")

        plot_placeholder_2.plotly_chart(
            fig_convolucion, use_container_width=True, key=f"convolution_chart_{shift}"
        )
        time.sleep(0.5)  # Pausa para la animación


# Configuración inicial de la aplicación en Streamlit
st.set_page_config(layout="wide")
st.title("Interfaz gráfica de convolución de señales")

st.sidebar.title("Menu de operaciones")
operation = st.sidebar.selectbox(
    "Tipo de Señal", ["Menú Inicial...", "Continua", "Discreta"]
)

# Menú inicial
if operation == "Menú Inicial...":
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

# Operaciones con señales continuas
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

        inv = st.sidebar.selectbox(
            "Cual señal desea invertir", ["Seleccione", "x(t)", "h(t)"]
        )
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
                convolucion_continua(h, z, x, y)
            elif inv == "h(t)":
                t_inv, x_inv = invertir(h, z)
                grafica_continua(t_inv, x_inv, "green", "h(t) invertida")
                st.markdown("### Proceso de convolución ###")
                convolucion_continua(x, y, h, z)

# Operaciones con señales discretas
elif operation == "Discreta":
    punto = st.sidebar.selectbox("Señal x[n]", ["Seleccione", "A", "B"])
    if punto == "Seleccione":
        st.error("Seleccione las señales a graficar")
    else:
        if punto == "A":
            col1, col2 = st.columns(2)
            with col1:
                x = na
                y = xn_a
                graf = stem(na, xn_a, "x[n]", "blue")
                st.plotly_chart(graf, use_container_width=True)
            with col2:
                h = ha
                z = hn_a
                graf = stem(ha, hn_a, "h[n]", "red")
                st.plotly_chart(graf, use_container_width=True)

        elif punto == "B":
            col1, col2 = st.columns(2)
            with col1:
                x = nb
                y = xn_b
                graf = stem(nb, xn_b, "x[n]", "blue")
                st.plotly_chart(graf, use_container_width=True)
            with col2:
                h = hb
                z = hn_b
                graf = stem(hb, hn_b, "h[n]", "red")
                st.plotly_chart(graf, use_container_width=True)

        invertir = st.sidebar.selectbox(
            "Cual señal desea invertir", ["Seleccione", "x[n]", "h[n]"]
        )
        if invertir == "Seleccione":
            st.error("Seleccione la señal a invertir")
        else:
            if invertir == "x[n]":
                x_inv, y_inv = invertir_discreta(x, y)
                inv = stem(x_inv, y_inv, "x[n] invertida", "green")
                st.plotly_chart(inv, use_container_width=True)

                convolucion_discreta(h, x, z, y)
            elif invertir == "h[n]":
                h_inv, z_inv = invertir_discreta(h, z)
                inv = stem(h_inv, z_inv, "h[n] invertida", "green")
                st.plotly_chart(inv, use_container_width=True)
                convolucion_discreta(x, h, y, z)
