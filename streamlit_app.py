# Importamos las librerias necesarias
import numpy as np
import plotly.graph_objs as go
import streamlit as st
import time

# Datos originales
Delta = 0.01

# Tiempo total
ta = [0]
ta2 = np.arange(0, 3, Delta)
ta3 = np.arange(3, 5, Delta)
ta4 = [5]
ta_T = np.concatenate((ta, ta2, ta3, ta4))

# Valores de la función
a = [0]
xa2 = 2 * np.ones(len(ta2))
xa3 = -2 * np.ones(len(ta3))
xa4 = [0]
x_ta = np.concatenate((a, xa2, xa3, xa4))

# Invertir los valores del eje X y los valores correspondientes del eje Y
ta_T_inverted = -ta_T[::-1]
x_ta_inverted = x_ta[::-1]


def graficar(ta, x_ta):
    plot_placeholder = st.empty()

    # Crear la figura inicial
    trace = go.Scatter(x=ta, y=x_ta, mode='lines', name='Señal Invertida')
    layout = go.Layout(xaxis=dict(range=[-21, 21]), yaxis=dict(range=[-3, 3]), title='Señal Invertida en Movimiento')
    fig = go.Figure(data=[trace], layout=layout)

    # Actualizar la señal para moverla de izquierda a derecha en el eje X
    for shift in np.linspace(-20, 20, 20):
        # Actualizar los valores de X para moverlos
        new_x = ta_T_inverted + shift
        fig.data[0].x = new_x
        
        # Renderizar la figura actualizada en el mismo lienzo
        plot_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Agregar un pequeño retardo para ver la animación
        time.sleep(0.3)


st.set_page_config(layout="wide")
st.title("Interfaz gráfica de procesamiento de señales")

st.sidebar.title("Menu de operaciones")
operation = st.sidebar.selectbox(
    "Tipo de Señal", ["Menú Inicial...", "Continua", "Discreta"]
)

if operation == "Menú Inicial...":
    st.markdown("Este laboratorio tiene como objetivo poner en práctica los conceptos teóricos aprendidos en el curso de Señales y Sistemas. Aquí tendrás la oportunidad de realizar operaciones fundamentales de transformación de señales y visualizar los resultados en un entorno gráfico computacional.")
    st.markdown("Para comenzar a usar la aplicación, dirígete a la parte izquierda y selecciona el menú desplegable debajo del menú de operaciones. Desde allí podrás escoger el tipo de señal con la que deseas trabajar y también volver a esta pestaña cuando lo desees.")
    st.subheader("Explicación Teorica de las transformaciones")
    st.markdown("La transformación de señales se refiere a los cambios que pueden producirse en los parámetros que componen una señal. Hay dos tipos principales de transformaciones: el escalamiento y el desplazamiento. Es importante analizar por separado las transformaciones en tiempo continuo y en tiempo discreto debido a las diferencias significativas entre ambos casos.")
    st.markdown("##### Desplazamiento en el tiempo #####")
    st.markdown(r"Una señal $x(t)$ se desplaza en el tiempo cuando la variable $t$ en dicha señal se suma o se resta por un valor $t_0$. El desplazamiento es hacia la derecha, o un retardo, si es $-t_0$, y hacia la izquierda, o un adelanto, si es $+t_0$. Si la señal experimenta una reflexión, es decir, cuando la variable t se multiplica por menos uno $x(-t)$, los desplazamientos ocurren en la dirección opuesta.")
    st.markdown("##### Escalamiento en el tiempo #####")
    st.markdown(r"Una señal $x(t)$ se escala en el tiempo cuando la variable t es multiplicada por un valor absoluto $|a|$ mayor o menor que uno, es decir, $x(at)$. Si $|a|$ es mayor que 1, la señal se comprime por ese factor; si es menor que 1, se expande por el mismo factor. Cada valor de $t$ se multiplica por el inverso de $a$ para obtener los nuevos valores.")
    st.markdown(r"Hay dos formas de transformar una señal $x(t)$ en el tiempo: primero desplazarla y luego escalarla, o realizar el proceso en el orden inverso.")
    st.sidebar.markdown("Creado por: Oliver Ardila, Juan Bermejo y Daniel Henriquez")

    st.markdown("#### Agradecimientos ####")
    col1, col2 = st.columns(2)
    with col1:
        st.image("aura.jpg", width=400)
    with col2:
        st.image("lena.png", width=400)

    st.markdown("Damos gracias a Cristiano Ronaldo y a Lena Gray por el desarrollo de esta interfaz gráfica.")
elif operation == "Continua":
    signal = st.sidebar.radio("Señal", ["1", "2"])
    graficar(ta_T_inverted, x_ta_inverted)
