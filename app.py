import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import accuracy_score

st.title('Breast Cancer Wisconsin')
st.write("## Verificação se o câncear é maligno ou benigno")

st.sidebar.write("### Parâmetros 0")

radius0 = st.sidebar.slider("Radius", 6.981000, 28.110000, 0.1)
texture0 = st.sidebar.slider("Texture", 9.710000, 39.28000, 0.1)
smoothness0 = st.sidebar.slider("Smoothness", 0.052630, 0.16340, 0.1)
concave_points0 = st.sidebar.slider("Concave points", 0.0, 0.20120, 0.1)
symmetry0 = st.sidebar.slider("Symmetry", 0.106000, 0.30400, 0.1)
fractal_dimension0 = st.sidebar.slider("Fractal dimension", 0.049960, 0.09744, 0.1)

st.sidebar.write("### Parâmetros 1")

radius1 = st.sidebar.slider("Radius", 0.111500, 2.87300, 0.1)
texture1 = st.sidebar.slider("Texture", 0.360200, 4.88500, 0.1)
smoothness1 = st.sidebar.slider("Smoothness", 0.001713, 0.03113, 0.1)
concave_points1 = st.sidebar.slider("Concave points", 0.0, 0.05279, 0.1)
symmetry1 = st.sidebar.slider("Symmetry", 0.007882, 0.07895, 0.1)
fractal_dimension1 = st.sidebar.slider("Fractal dimension", 0.000895, 0.02984, 0.1)

st.sidebar.write("### Parâmetros 2")

radius2 = st.sidebar.slider("Radius", 7.930000, 36.04000, 0.1)
texture2 = st.sidebar.slider("Texture", 12.020000, 49.54000, 0.1)
smoothness2 = st.sidebar.slider("Smoothness", 0.071170, 0.22260, 0.1)
concave_points2 = st.sidebar.slider("Concave points", 0.0, 0.29100, 0.1)
symmetry2 = st.sidebar.slider("Symmetry", 0.156500, 0.66380, 0.1)
fractal_dimension2 = st.sidebar.slider("Fractal dimension", 0.055040, 0.20750, 0.1)

arquivo = open("objetos.pkl", "rb")
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
