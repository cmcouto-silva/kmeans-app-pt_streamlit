import kmeans
import templates

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

import streamlit as st	
import streamlit.components.v1 as components

import plotly.graph_objects as go

st.set_page_config(page_title="K-means Visualization App", page_icon="python.png", layout="wide")

st.markdown("""
**Autor:** Cainã Max Couto da Silva  
**LinkedIn:** [cmcouto-silva](https://www.linkedin.com/in/cmcouto-silva/)
""")

st.header('**Visualizando o algorimo K-means etapa por etapa com Python**')
st.write("")

# components.html('<b>texto</b>')
# st.markdown("<h1 style='text-align: center; color: red;'>texto</h1>", unsafe_allow_html=True)

st.sidebar.title('Parâmetros')

with st.sidebar.beta_container():
   _, slider_col, _ = st.beta_columns([0.02, 0.96, 0.02])
   with slider_col:
        k = st.sidebar.select_slider(
				label='Número de grupos (clusters) simulados:',
				options=range(2,11), value=2
			)
        std = st.sidebar.slider(
				'Desvio padrão dos dados:',
				0.1, 5.0, 1.0, 0.1
			)

mode = st.sidebar.selectbox('Modo de inicialização dos clusters:', ["random", "kmeans++"])

_, central_button, _ = st.sidebar.beta_columns([0.25, 0.5, 0.25])
with central_button:
	st.text("")
	st.button('Recomputar')

st.markdown("""
Para melhor compreensão da técnica, visualize seu passo a passo abaixo.

Experimente modificar a quantidade de amostras, clusters e desvio padrão das amostras. Observe que, em geral, 
quanto menor o número de grupos e menor o desvio padrão da amostra, mais rápido se alcança a convergência da posição final dos centroides.
O algoritmo kmeans++ também tende a acelerar a convergência dos centroides.  

""")

with st.beta_expander(label='Ler explicação do método'):
	
	st.markdown("""
	K-means é um modelo de machine learning não supervisionado que visa identificar grupos (*i.e.* clusters) no conjunto de dados. Assim como todo modelo não supervisionado, seu objetivo é **identificar padrões nos dados** e interpretá-los, **não sendo adequado para predição**.

	As variáveis utilizadas neste modelo são, **necessariamente**, numéricas. Como resultado, identificamos cada amostra à um grupo. A quantidade de grupos (K) identificados é determinada _a priori_, ou seja, antes de rodar o modelo. Contudo, existem métodos, como o método do cotovelo e da silhueta (Elbow e Silhouette, respectivamente), que nos auxiliam a identificar a melhor quantidade de grupos para nossos dados.

	As variáveis categóricas podem ser utilizas após a descrição e interpretação das amostras presentes nestes grupos. **Não é recomendado** o uso de [ponderação arbitrária]() para transformar variáveis categóricas em variáveis numéricas. Ao transformar variáveis categóricas em variáveis numéricas sem respaldo da literatura, assume-se que a distância entre as categorias são as mesmas, que por sua vez pode introduzir um viés na análise.

	O modelo K-means é essencialmente baseado no cálculo da distância - normalmente a [distância euclidiana]() - entre os pontos da amostra e os centroides, que são os pontos inicialmente aleatórios e representam o centro de cada grupo (cluster). Assim, a quantidade de centroides é igual ao número de grupos (estes representados pela letra "k").

	Cabe ressaltar que, uma vez que K-means utiliza a distância como métrica do modelo, é necessário normalizar os dados caso as variáveis estejam em escalas diferentes.

	**Em resumo, a técnica funciona assim:**

	1. Definir um número de clusters (k)
	2. Inicializar os k centroides
	3. Categorizar cada ponto ao seu centroide mais próximo
	4. Mover os centroides para o centro (média) dos pontos em sua categoria
	5. Repetir as etapas 3 e 4 até as posições dos centroides não modificaram ou atingir um número máximo de iterações.  

	&nbsp;

	A inicialização dos centroides pode ser totalmente aleatória ou otimizada utilizando um algoritmo conhecido como K-means++.

	Na inicialização aleatória, selecionamos k pontos pré-existentes como centroides, ou atribuímos k centroides dentro da dimensão dos pontos dos nossos dados. Na técnica K-means++ o objetivo é inicializar os centroides o mais distante possível um do outro, que por sua vez pode eliminar viés de inicialização de centroides (vide tópico "Armadilha do K-means") e tende a diminuir a quantidade de iterações necessárias para convergência da posição final dos centroides.

	Adicionalmente, algumas aplicações também rodam mais de uma vez o algoritmo com diferentes pontos de inicialização dos centroides, fornecendo como output aquele com a menor soma das variâncias internas de cada grupo.
	
	_**Observações:**_

	---

	Os dados desta aplicação são simulados com a função [`datasets.make_blobs()`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html) da biblioteca [`scikit-learn`](https://scikit-learn.org/stable/index.html). Aqui, limitei tanto a quantidade de observações (100) quanto de variáveis (2), para a aplicação não ficar pesada e ser possível visualizar com gráficos bidimensionais, respectivamente. 

	Não foi utilizado mais de uma corrida por k, de modo que isso pode influenciar no método do cotovelo, visto que para determinados Ks a inicialização do centroide não foi a das melhores. 

	Sugestões ou críticas? Contate-me via [LinkedIn](https://www.linkedin.com/in/cmcouto-silva/).

	_**Disponibilização dos códigos:**_

	---

	Os scripts com a implementação passo a passo do K-means e produção deste aplicativo estão disponíveis no GitHub:

	- [Repositório do aplicativo](https://github.com/cmcouto-silva/kmeans-app-pt_streamlit)
	- [Script K-means passo a passo (sem scikit-learn)](https://github.com/cmcouto-silva/kmeans-app-pt_streamlit/blob/main/kmeans.py)

	&nbsp;
	""")

data = make_blobs(centers=k, cluster_std=std)
df = pd.DataFrame(data[0], columns=['x','y']).assign(label = data[1])

if st.checkbox('Mostrar dados brutos'):
	_, df_col, _ = st.beta_columns([0.25,0.2, 0.25])
	with df_col:
		st.write(df)

model, wss = kmeans.calculate_WSS(data[0], k, 10, mode=mode)
raw_col, elbow_col = st.beta_columns([0.5,0.5])

_, kanimation_col, _ = st.beta_columns([0.2,0.8,0.2])

with kanimation_col:
	fig = kmeans.plot(model)
	fig = fig.update_layout(autosize=False, height=560,
		title_text="<b>Visualizando as etapas do K-means</b>", title_font=dict(size=24))
	st.plotly_chart(fig, use_container_width=True, sharing="streamlit")

with raw_col:
	raw_fig = go.Figure(
		data=fig.data[0],
		layout=dict(
			template='seaborn', title='<b>Pontos sem agrupamento:</b>',
			xaxis=dict({'title':'x'}), yaxis=dict({'title':'y'})
			)
		)
	st.plotly_chart(raw_fig)

with elbow_col:
	elbow_fig = go.Figure(
	data=go.Scatter(x=list(range(1,11)), y=wss),
	layout=dict(
		template='seaborn', title='<b>Método do cotovelo</b>',
		xaxis=dict({'title':'k'}), yaxis=dict({'title':'wss'})
		)
	)
	st.plotly_chart(elbow_fig)


st.markdown("""
---

## **Armadilha do K-means**

A inicialização aleatória dos centroides podem criar grupos que não representam grupos reais.
No exemplo abaixo, percebe-se claramente a existência de quatro grupos (à esquerda), 
equanto na animação à direita é possível observar que a iniciando os centroides próximos um dos outros,
neste caso, levou a um agrupamento não fidedigno dos dados reais. Por este motivo normalmente se utiliza `kmeans++` e/ou
aplica algumas vezes o modelo, ficando com aquele com menos distorção. 

Por fim, é importante salientar que há outras estruturas de agrupamentos mais complexas
onde o modelo K-means não possui bom desempenho. Nestes casos, usa-se outros modelos de agrupamento.
Como exemplo, a primeira figura da [página de modelos de clusterização](https://scikit-learn.org/stable/modules/clustering.html) do scikit-learn 
ilustra bem como cada modelo se sai na identificação de grupos em dados com diferentes formatos (distribuições).

&nbsp;

""")

# Specific biased data
raw_seed, kanimation_seed = st.beta_columns([0.5,0.5])

data_seed,labels_seed = make_blobs(centers=4, random_state=3)
model_seed = kmeans.Kmeans(data_seed, 4, seed=2)
model_seed.fit()

with raw_seed:
	raw_fig_seed = go.Figure(
		data=go.Scatter(x=data_seed[:,0], y=data_seed[:,1], mode='markers', marker=dict(color=labels_seed)),
		layout=dict(title_text="<b>Pontos coloridos conforme os grupos reais</b>",
			template="simple_white", title_font=dict(size=18)))
	raw_fig_seed.update_layout(templates.simple_white, height=500, title_x=0.15, title_font_size=18)
	st.plotly_chart(raw_fig_seed, use_container_width=True, sharing="streamlit")

with kanimation_seed:
	fig_seed = kmeans.plot(model_seed)
	fig_seed = fig_seed.update_layout(autosize=False, height=500,
		title_text="<b>Visualizando viés de inicialização dos centroides</b>", title_font=dict(size=24))
	st.plotly_chart(fig_seed, use_container_width=True, sharing="streamlit")


# st.markdown("""Para sugestões e críticas, favor me contatar pelo [LinkedIn](https://www.linkedin.com/in/cmcouto-silva/).""")
