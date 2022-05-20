import streamlit as st
#import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import accuracy_score
from PIL import Image
#from gsheetsdb import connect

  
st.title('Relatórios Breast Cancer Wisconsin')
st.header("Informações do conjunto de dados:")
st.write(
    """As características são computadas a partir de uma imagem digitalizada de um aspirado
    por agulha fina (PAAF) de uma massa mamária.Eles descrevem características 
    dos núcleos celulares presentes na imagem..""" )

IMAGE_URL = "https://miro.medium.com/max/1400/1*yjsLGG-U9km84AvWLLmK8A.png"
st.image(IMAGE_URL, caption="Imagem da célula cancerosa")

st.header("Conjunto de dados:")
IMAGE_URL = "https://miro.medium.com/max/1400/1*51Hm0b9RlgnPVQLariliRw.png"
st.image(IMAGE_URL, caption="Sunrise by the mountains")

st.sidebar.write("### Parâmetros 0")

radius0 = st.sidebar.slider("Radius 0", 7.0, 29.0, 0.1)
texture0 = st.sidebar.slider("Texture 0", 9.8, 40.0, 0.1)
smoothness0 = st.sidebar.slider("Smoothness 0", 0.06, 0.17, 0.1)
concave_points0 = st.sidebar.slider("Concave points 0", 0.0, 0.10, 0.1)
symmetry0 = st.sidebar.slider("Symmetry 0", 0.2, 0.31, 0.1)
fractal_dimension0 = st.sidebar.slider("Fractal dimension 0", 0.05, 0.10, 0.1)

st.sidebar.write("### Parâmetros 1")

radius1 = st.sidebar.slider("Radius 1", 0.2, 3.3, 0.1)
texture1 = st.sidebar.slider("Texture 1", 0.4, 5.0, 0.1)
smoothness1 = st.sidebar.slider("Smoothness 1", 0.002, 0.032, 0.1)
concave_points1 = st.sidebar.slider("Concave points 1", 0.0, 0.053, 0.1)
symmetry1 = st.sidebar.slider("Symmetry 1", 0.008, 0.079, 0.1)
fractal_dimension1 = st.sidebar.slider("Fractal dimension 1", 0.0009, 0.030, 0.1)

st.sidebar.write("### Parâmetros 2")

radius2 = st.sidebar.slider("Radius 2", 8.0, 37.0, 0.1)
texture2 = st.sidebar.slider("Texture 2", 13.0, 50.0, 0.1)
smoothness2 = st.sidebar.slider("Smoothness 2", 0.08, 0.23, 0.1)
concave_points2 = st.sidebar.slider("Concave points 2", 0.0, 0.30, 0.1)
symmetry2 = st.sidebar.slider("Symmetry 2", 0.2, 0.67, 0.1)
fractal_dimension2 = st.sidebar.slider("Fractal dimension 2", 0.06, 0.21, 0.1)
   
with open("objetos.pkl", "rb") as arquivo:
  ss, classifier = pickle.load(arquivo)
  
  estrutura = {'0radius' : radius0, '0texture': texture0, '0smoothness' : smoothness0, '0concave points': concave_points0,
               '0symmetry' : symmetry0, '0fractal dimension' : fractal_dimension0, '1radius' : radius1, '1texture': texture1,
               '1smoothness' : smoothness1, '1concave points': concave_points1, '1symmetry' : symmetry1, '1fractal dimension' : fractal_dimension1,
               '2radius' : radius2, '2texture': texture2, '2smoothness' : smoothness2, '2concave points': concave_points2,
               '2symmetry' : symmetry2, '2fractal dimension' : fractal_dimension2}
  df = pd.DataFrame(estrutura, index=[0])
 
  st.write("### Parâmetros de Entrada")
  st.write(df)
  
  
  st.dataframe(df)
  df_sample = df.head()
  df_sample
    
  df = ss.transform(df)
  st.write(df)
  
  predicao = classifier.predict(df)
  st.write(f"A classe é: **{predicao[0]}**")
  
  predicao = classifier.predict_proba(df)
  predicao = pd.DataFrame(predicao)
  predicao.rename({
     'M' : 0,
     'B' : 1
  }, axis=1, inplace=True)
  
  st.write("Probabilidades")
  st.write(predicao)
  
  dataframe = pd.DataFrame(np.random.randn(10, 20),
  columns = ('col %d' % i
    for i in range(20)))
  st.write(dataframe)
  st.header('Visualização do gráfico de área.')
  st.area_chart(dataframe)
  #st.header('Visualização do histograma.')
  #st.bar_chart(dataframe)
    
   
  st.write("### Informações do atributo:")
  st.write( """ a. perímetro(soma dos tamanhos dos lados da figura)""" )
  st.write( """ b. área (medida total que uma figura ocupa no plano)""" )
  st.write( """ c. compacidade (perímetro^2 / área - 1,0)""" )
  st.write( """ d. concavidade (severidade das porções côncavas do contorno).""" )
