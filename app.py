import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time

# Vectores dados
x = [1, 2, 3, 4]
y = [2, 3, 1, 4]
h = [1, 3, 4, 2]
z = [4, 1, 3, 2]

# Crear gráfico estático usando una gráfica stem
fig_static = go.Figure()

fig_static.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='y'))
fig_static.add_trace(go.Scatter(x=x, y=h, mode='lines+markers', name='h'))

fig_static.update_layout(title='Gráfica Estática', xaxis_title='X', yaxis_title='Y/H')

# Mostrar gráfica estática en Streamlit
st.plotly_chart(fig_static)

# Gráfica que se mueve (animación)
fig_dynamic = go.Figure()

# Agregar el primer frame de la gráfica
fig_dynamic.add_trace(go.Scatter(x=x, y=z, mode='lines+markers', name='z'))

fig_dynamic.update_layout(title='Gráfica Dinámica', xaxis_title='X', yaxis_title='Z')

# Actualizar la gráfica en tiempo real
dynamic_plot = st.empty()  # Crear un placeholder para la gráfica dinámica

for i in range(10):
    z = np.random.randint(1, 5, size=len(x))  # Simulación de nuevos datos para z
    fig_dynamic.data = []  # Limpiar la figura antes de agregar nuevos datos
    fig_dynamic.add_trace(go.Scatter(x=x, y=z, mode='lines+markers', name='z'))
    dynamic_plot.plotly_chart(fig_dynamic)  # Actualizar el gráfico en la app
    time.sleep(1)  # Pausa de 1 segundo para simular actualización
