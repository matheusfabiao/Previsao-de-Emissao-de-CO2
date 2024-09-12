# Documentação do Projeto: Previsão de Emissão de CO2 com Regressão Linear

Este repositório contém um projeto simples de aprendizado de máquina que utiliza **Regressão Linear** para prever a emissão de gás carbônico (CO2) com base no tamanho do motor de veículos. O projeto é implementado em um **Jupyter Notebook**, utilizando a biblioteca **scikit-learn** e outros pacotes populares de manipulação e visualização de dados.

## Objetivo

O objetivo deste projeto é demonstrar como construir um modelo de Regressão Linear para prever a emissão de CO2 de veículos com base na variável "tamanho do motor" presente em um dataset. Vamos dividir os dados em conjuntos de treino e teste, treinar o modelo e avaliar seu desempenho usando métricas comuns.

## Estrutura do Repositório

- **data**: Diretório onde será colocado o dataset `FuelConsumptionCo2.csv`.
- **notebooks**: Contém o arquivo `.ipynb` com o código para realizar a análise.
- **README.md**: Este arquivo de documentação.

## Pré-requisitos

Antes de começar, você precisará instalar os seguintes pacotes:

- Python 3.x
- Jupyter Notebook
- Bibliotecas necessárias:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Dataset

O dataset usado neste projeto é chamado `FuelConsumptionCo2.csv` e contém informações sobre o consumo de combustível e emissões de CO2 de veículos. As principais colunas de interesse são:

- **ENGINESIZE**: Tamanho do motor.
- **CO2EMISSIONS**: Emissão de CO2 do veículo.

## Código

### Importação de Bibliotecas

O projeto utiliza as seguintes bibliotecas:

```python
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt
import warnings
```

### Carregamento e Visualização dos Dados

Carregamos o dataset e visualizamos suas primeiras linhas:

```python
df = pd.read_csv('FuelConsumptionCo2.csv')
df.head()
```

### Separação das Features

Selecionamos as colunas de interesse: `ENGINESIZE` (tamanho do motor) e `CO2EMISSIONS` (emissão de CO2):

```python
engines = df[['ENGINESIZE']]
co2 = df[['CO2EMISSIONS']]
```

### Divisão dos Dados

Dividimos o dataset em um conjunto de treinamento (80%) e um conjunto de teste (20%):

```python
engines_train, engines_test, co2_train, co2_test = train_test_split(engines, co2, test_size=0.2, random_state=42)
```

### Visualização da Correlação

Plotamos a relação entre o tamanho do motor e a emissão de CO2 no conjunto de treino:

```python
plt.scatter(engines_train, co2_train, color='blue')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emissions')
plt.show()
```

### Treinamento do Modelo de Regressão Linear

Criamos e treinamos um modelo de Regressão Linear simples:

```python
model = linear_model.LinearRegression()
model.fit(engines_train, co2_train)
```

Exibimos os coeficientes da equação linear:

```python
print('(A) Intercept: ', model.intercept_)  
print('(B) Slope: ', model.coef_)  
```

### Avaliação e Previsão

Realizamos previsões no conjunto de teste e avaliamos o modelo usando várias métricas:

```python
co2Predictions = model.predict(engines_test)
```

Avaliação do modelo com métricas como MSE, RMSE, MAE e R2-score:

```python
print("MSE: %.2f" % mean_squared_error(co2_test, co2Predictions))
print("RMSE: %.2f " % sqrt(mean_squared_error(co2_test, co2Predictions)))
print("MAE: %.2f" % mean_absolute_error(co2_test, co2Predictions))
print("R2-score: %.2f" % r2_score(co2_test, co2Predictions))
```

### Visualização dos Resultados

Plotamos as previsões e a linha de regressão:

```python
plt.scatter(engines_test, co2_test, color='blue')
plt.plot(engines_test, model.coef_[0][0]*engines_test + model.intercept_[0], '-r')
plt.ylabel('CO2 Emission')
plt.xlabel('Engines')
plt.show()
```

## Avaliação do Modelo

Métricas utilizadas para avaliar o desempenho do modelo:

- **Erro Quadrático Médio (MSE)**: Mede a média dos erros quadráticos entre os valores reais e previstos.
- **Raiz do Erro Quadrático Médio (RMSE)**: A raiz quadrada do MSE.
- **Erro Médio Absoluto (MAE)**: Representa a média dos erros absolutos entre os valores reais e previstos.
- **R2-score**: Avalia o quão bem o modelo consegue prever os dados (quanto mais próximo de 1, melhor).

## Conclusão

Este projeto mostra como é possível utilizar a Regressão Linear para prever a emissão de CO2 com base no tamanho do motor de um veículo. O uso de métricas como MSE e R2-score permite uma análise quantitativa da precisão do modelo.

## Como Contribuir

Se quiser contribuir, sinta-se à vontade para fazer um fork do projeto e enviar suas sugestões através de pull requests.

## Licença

Este projeto está licenciado sob os termos da licença MIT.

---

**Autor**: Matheus Fabião
