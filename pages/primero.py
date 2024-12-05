# Importamos las librerías necesarias
import numpy as np
import sympy as sp
from sympy import *
import streamlit as st
import plotly.graph_objs as go

delta=0.01

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
        xaxis_title="Armónico",
        yaxis_title="Amplitud",
        showlegend=True,
        template="plotly_white",
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig

# Configuración inicial de la aplicación en Streamlit
st.set_page_config(layout="wide")
st.title("Ejemplo de Series")

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


st.subheader("Seleccione el ejemplo")
operation = st.selectbox(
    "", ["1", "2", "3", "4"]
)

if operation:
  st.subheader("Inserte el numero de armonicos")
  N = st.number_input(
    "", value=None, placeholder="Digite un número...")
  if N:
    N = int(N)
    if operation == "1":
      T=4    #Definir el periodo de esta señal
      t1 = np.arange(-T/2, 0, delta)
      t2 = np.arange(0, T/2 + delta, delta)
      x1t = 1 + 4 * (t1 / T)
      x2t = 1 - 4 * (t2 / T)

      x_ciclo = np.concatenate((x1t, x2t))
      t_ciclo = np.concatenate((t1, t2))

      periodo = 2  # Número de periodos a repetir
      t_periodo = []
      x_periodo= []
      for k in range(-periodo // 2, periodo // 2 + 1):
          t_periodo.append(t_ciclo + k * T)  # Desplazar tiempo por múltiplos de T
          x_periodo.append(x_ciclo)  # Repetir el mismo ciclo

      t_periodo = np.concatenate(t_periodo)
      x_periodo = np.concatenate(x_periodo)
      t= sp.Symbol('t')
      n= sp.Symbol('n')
      x=sp.Symbol('x')
      T=4
      f1=1+4*(t/T)
      f2=1-4*(t/T)
      af1=(2/T)*f1*sp.cos(n*(2*sp.pi/T)*t)
      af2=(2/T)*f2*sp.cos(n*(2*sp.pi/T)*t)
      aI= integrate(af1,(t,-T/2,0))+integrate(af2,(t,0,T/2))
      a0= aI.subs(n,0).evalf()
      bf1=(2/T)*f1*sp.sin(n*(2*sp.pi/T)*t)
      bf2=(2/T)*f2*sp.sin(n*(2*sp.pi/T)*t)
      bI= integrate(bf1,(t,-T/2,0))+integrate(bf2,(t,0,T/2))

      xn=0
      for i in range(N):
        if(i==0):
          xn+=a0
        else:
          xn+= aI.subs(n,i).evalf()*cos(x*2*i*sp.pi/T)+bI.subs(n,i).evalf()*sin(x*2*i*sp.pi/T)


      li = t_periodo[0]
      lf = t_periodo[-1]
      delta = 0.01
      tn = np.arange(li, lf+delta, delta)  # Crear puntos para el eje x


      f_lambda = sp.lambdify(x, xn, modules=["numpy"])  # Convertir función simbólica a una evaluable

      Cn=sqrt(aI**2+bI**2)
      Sp=np.zeros(N+N+1)
      ts=np.arange(-N,N+1)
      k=0
      for i in range (-N,N+1):
        if(i==0):
          Sp[k]=a0/2
        else:
          Sp[k]=Cn.subs(n,i).evalf()
        k=k+1
    elif operation == "2":
      T=2*np.pi  #Definir el periodo de esta señal
      t1b = np.arange(-T/2, 0, delta)
      t2b = np.arange(0, T/2 + delta, delta)
      x1tb = t1b
      x2tb = t2b

      x_ciclo = np.concatenate((x1tb, x2tb))
      t_ciclo = np.concatenate((t1b, t2b))

      periodo = 2  # Número de periodos a repetir
      t_periodo = []
      x_periodo= []
      for k in range(-periodo // 2, periodo // 2 + 1):
          t_periodo.append(t_ciclo + k * T)  # Desplazar tiempo por múltiplos de T
          x_periodo.append(x_ciclo)  # Repetir el mismo ciclo

      t_periodo = np.concatenate(t_periodo)
      x_periodo = np.concatenate(x_periodo)

      t= sp.Symbol('t')
      n= sp.Symbol('n')
      x=sp.Symbol('x')
      T=2*np.pi
      f1=t
      af1=(2/T)*f1*sp.cos(n*(2*np.pi/T)*t)
      aI= integrate(af1,(t,-np.pi,np.pi))
      a0=aI.subs(n,0).evalf()
      bf1=(2/T)*f1*sp.sin(n*(2*np.pi/T)*t)
      bI= integrate(bf1,(t,-np.pi,np.pi))

      xn=0
      for i in range(N):
        if(i==0):
          xn+=a0
        else:
          xn+= aI.subs(n,i).evalf()*cos(x*2*i*sp.pi/T)+bI.subs(n,i).evalf()*sin(x*2*i*sp.pi/T)


      li = t_periodo[0]
      lf = t_periodo[-1]
      delta = 0.01
      tn = np.arange(li, lf+delta, delta)  # Crear puntos para el eje x


      f_lambda = sp.lambdify(x, xn, modules=["numpy"])  # Convertir función simbólica a una evaluable

      Cn=sqrt(aI**2+bI**2)
      Sp=np.zeros(N+N+1)
      ts=np.arange(-N,N+1)
      k=0
      for i in range (-N,N+1):
        if(i==0):
          Sp[k]=a0/2
        else:
          Sp[k]=Cn.subs(n,i).evalf()
        k=k+1
    elif operation == "3":
      T=2*np.pi  #Definir el periodo de esta señal
      t1b = np.arange(-T/2, 0, delta)
      t2b = np.arange(0, T/2 + delta, delta)
      x1tb = t1b**2
      x2tb = t2b**2

      x_ciclo = np.concatenate((x1tb, x2tb))
      t_ciclo = np.concatenate((t1b, t2b))

      periodo = 2  # Número de periodos a repetir
      t_periodo = []
      x_periodo= []
      for k in range(-periodo // 2, periodo // 2 + 1):
          t_periodo.append(t_ciclo + k * T)  # Desplazar tiempo por múltiplos de T
          x_periodo.append(x_ciclo)  # Repetir el mismo ciclo

      t_periodo = np.concatenate(t_periodo)
      x_periodo = np.concatenate(x_periodo)

      t= sp.Symbol('t')
      n= sp.Symbol('n')
      x=sp.Symbol('x')
      T=2*np.pi
      f1=(t)**2
      af1=(2/T)*f1*sp.cos(n*(2*np.pi/T)*t)
      aI= integrate(af1,(t,-np.pi,np.pi))
      a0= aI.subs(n,0).evalf()
      bf1=(2/T)*f1*sp.sin(n*(2*np.pi/T)*t)
      bI= integrate(bf1,(t,-np.pi,np.pi))

      xn=0
      for i in range(N):
        if(i==0):
          xn+=a0
        else:
          xn+= aI.subs(n,i).evalf()*cos(x*2*i*sp.pi/T)+bI.subs(n,i).evalf()*sin(x*2*i*sp.pi/T)


      li = t_periodo[0]
      lf = t_periodo[-1]
      delta = 0.01
      tn = np.arange(li, lf+delta, delta)  # Crear puntos para el eje x


      f_lambda = sp.lambdify(x, xn, modules=["numpy"])  # Convertir función simbólica a una evaluable

      Cn=sqrt(aI**2+bI**2)
      Sp=np.zeros(N+N+1)
      ts=np.arange(-N,N+1)
      k=0
      for i in range (-N,N+1):
        if(i==0):
          Sp[k]=a0/2
        else:
          Sp[k]=Cn.subs(n,i).evalf()
        k=k+1
      
    elif operation == "4":
      T=2  #Definir el periodo de esta señal
      t1b = np.arange(-T/2, 0, delta)
      t2b = np.arange(0, T/2 + delta, delta)
      x1tb = t1b
      x2tb = np.ones(len(t2b))
      x_ciclo = np.concatenate((x1tb, x2tb))
      t_ciclo = np.concatenate((t1b, t2b))

      periodo = 2  # Número de periodos a repetir
      t_periodo = []
      x_periodo= []
      for k in range(-periodo // 2, periodo // 2 + 1):
          t_periodo.append(t_ciclo + k * T)  # Desplazar tiempo por múltiplos de T
          x_periodo.append(x_ciclo)  # Repetir el mismo ciclo

      t_periodo = np.concatenate(t_periodo)
      x_periodo = np.concatenate(x_periodo)

      t= sp.Symbol('t')
      n= sp.Symbol('n')
      x=sp.Symbol('x')
      T=2
      f1=t
      f2=1
      af1=(2/T)*f1*sp.cos(n*(2*sp.pi/T)*t)
      af2=(2/T)*f2*sp.cos(n*(2*sp.pi/T)*t)
      aI= integrate(af1,(t,-T/2,0))+integrate(af2,(t,0,T/2))
      a0= (2/T)*aI.subs(n,0).evalf()
      bf1=(2/T)*f1*sp.sin(n*(2*sp.pi/T)*t)
      bf2=(2/T)*f2*sp.sin(n*(2*sp.pi/T)*t)
      bI= integrate(bf1,(t,-T/2,0))+integrate(bf2,(t,0,T/2))

      xn=0
      for i in range(N):
        if(i==0):
          xn+=a0
        else:
          xn+= aI.subs(n,i).evalf()*cos(x*2*i*sp.pi/T)+bI.subs(n,i).evalf()*sin(x*2*i*sp.pi/T)


      li = t_periodo[0]
      lf = t_periodo[-1]
      delta = 0.01
      tn = np.arange(li, lf+delta, delta)  # Crear puntos para el eje x


      f_lambda = sp.lambdify(x, xn, modules=["numpy"])  # Convertir función simbólica a una evaluable

      Cn=sqrt(aI**2+bI**2)
      Sp=np.zeros(N+N+1)
      ts=np.arange(-N,N+1)
      k=0
      for i in range (-N,N+1):
        if(i==0):
          Sp[k]=a0/2
        else:
          Sp[k]=Cn.subs(n,i).evalf()
        k=k+1

    st.subheader("Numeros de armonicos: "+str(N))
    grafica_continua(t_periodo, x_periodo, "blue", "Señal Periódica")
    grafica_continua(tn, f_lambda(tn), "orange", "Señal Reconstruida")
    graf = stem(ts, Sp, "Espectro de Magnitud", "red")
    st.plotly_chart(graf, use_container_width=True)
