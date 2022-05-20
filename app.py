import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier  
import numpy as np

st.title('Breast Cancer Wisconsin')
st.write("## Verificação se o câncear é maligno ou benigno")

st.sidebar.write("### Parâmetros 0")

radius0 = st.sidebar.slider("Radius", 7.0, 29, 12.0, 0.1)
texture0 = st.sidebar.slider("Texture", 9.8, 40.0, 22.0, 0.1)
smoothness0 = st.sidebar.slider("Smoothness", 0.06, 0.17, 0.13, 0.1)
concave_points0 = st.sidebar.slider("Concave points", 0.0, 0.21, 0.10, 0.1)
symmetry0 = st.sidebar.slider("Symmetry", 0.2, 0.31, 0.20, 0.1)
fractal_dimension0 = st.sidebar.slider("Fractal dimension", 0.05, 0.10, 0.10, 0.1)

st.sidebar.write("### Parâmetros 1")

radius1 = st.sidebar.slider("Radius", 0.2, 3.3, 1.3, 0.1)
texture1 = st.sidebar.slider("Texture", 0.4, 5.0, 3.0, 0.1)
smoothness1 = st.sidebar.slider("Smoothness", 0.002, 0.032, 0.031, 0.1)
concave_points1 = st.sidebar.slider("Concave points", 0.0, 0.053, 0.040, 0.1)
symmetry1 = st.sidebar.slider("Symmetry", 0.008, 0.079, 0.050, 0.1)
fractal_dimension1 = st.sidebar.slider("Fractal dimension", 0.0009, 0.030, 0.029, 0.1)

st.sidebar.write("### Parâmetros 2")

radius2 = st.sidebar.slider("Radius", 8.0, 37.0, 22.0, 0.1)
texture2 = st.sidebar.slider("Texture", 13.0, 50.0, 10.0, 0.1)
smoothness2 = st.sidebar.slider("Smoothness", 0.08, 0.23, 0.12, 0.1)
concave_points2 = st.sidebar.slider("Concave points", 0.0, 0.30, 0.29, 0.1)
symmetry2 = st.sidebar.slider("Symmetry", 0.2, 0.67, 0.30, 0.1)
fractal_dimension2 = st.sidebar.slider("Fractal dimension", 0.06, 0.21, 0.10, 0.1)

arquivo = open('objetos.pkl', 'rb')
ss, classifier = pickle.load(arquivo)
arquivo.close()

estrutura = {'0radius' : radius0, '0texture': texture0, '0smoothness' : smoothness0, '0concave points': concave_points0,
               '0symmetry' : symmetry0, '0fractal dimension' : fractal_dimension0, '1radius' : radius1, '1texture': texture1,
               '1smoothness' : smoothness1, '1concave points': concave_points1, '1symmetry' : symmetry1, '1fractal dimension' : fractal_dimension1,
               '2radius' : radius2, '2texture': texture2, '2smoothness' : smoothness2, '2concave points': concave_points2,
               '2symmetry' : symmetry2, '2fractal dimension' : fractal_dimension2}
  
df = pd.DataFrame(estrutura, index=[0])
 
st.write("### Parâmetros de Entrada")
st.write(df)
  
df = ss.transform(df)
st.write(df)

previsoes = classifier.predict(df)
st.write(f"A classe é: **{previsoes[0]}**")
  
previsoes = classifier.predict_proba(df)
previsoes = pd.DataFrame(previsoes)
previsoes.rename({
    0 : 'Maligno',
    1 : 'Benigno'
}, axis=1, inplace=True)
  
st.write("Probabilidades")
st.write(previsoes)
