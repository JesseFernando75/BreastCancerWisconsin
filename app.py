import streamlit as st
#import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
#import plotly.express as px
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.title('Relatórios Breast Cancer Wisconsin')
st.write("## Classificação quanto ao Câncer")
#st.write("## Breast Cancer Wisconsin")

st.sidebar.write("### Parâmetros")
perimeter = st.sidebar.slider("Perimeter", 40.0, 190.0, 100.0)
area= st.sidebar.slider("Area", 184.0, 4254.0, 881.0)
compactness = st.sidebar.slider("Compactness", 0.01, 1.6, 0.25)
concavity = st.sidebar.slider("Concavity", 0.0, 1.30, 0.27)

    
with open("objetos.pkl", "rb") as arquivo:
  ss, classifier = pickle.load(arquivo)

  estrutura = {'perimeter': perimeter, 'area': area, 'compactness': compactness, 'concavity': concavity}
  df = pd.DataFrame(estrutura, index=[0])
 
  st.write("### Parâmetros de Entrada")
  st.write(df)
  
  
  #df = ss.transform(df)
  #st.write(df)
  
  #predicao = classifier.predict(df)
  #st.write(f"A classe é: **{predicao[0]}**")
  
  #predicao = classifier.predict_proba(df)
  #predicao = pd.DataFrame(predicao)
  #predicao.rename({
     #'M' : 0,
     #'B' : 1
  #}, axis=1, inplace=True)
  
  #st.write("Probabilidades")
  #st.write(predicao)
    
 # Visualização Gráfica
st.title('Visualização Gráfica')
# Grafico de correlção
plt.subplots(figsize=(5, 5)) 
sns.heatmap(train.corr(), annot=True, cmap='Blues')
ax.set_title('Correlação dos dados')
fig.tight_layout()
st.pyplot(fig)
    

 
