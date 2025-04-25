# Previsão de Demanda e Otimização de Fornecimento

## Descrição
Este projeto tem como objetivo prever a demanda de produtos com base em fatores como preço, estoque, descontos e sazonalidade, utilizando um modelo de **Regressão Linear**. Além disso, a otimização do fornecimento é realizada com a técnica de **Programação Linear**, buscando minimizar os custos de fornecimento dentro dos limites de estoque disponíveis.

## Tecnologias Utilizadas
- **Python**
- **pandas**: Para manipulação e análise de dados.
- **scikit-learn**: Para modelagem de Machine Learning e avaliação de desempenho.
- **scipy**: Para otimização de fornecimento utilizando programação linear.
- **matplotlib** e **seaborn**: Para visualização dos resultados.

## Funcionalidades

### 1. **Carregar e Limpar Dados**
A função `carregar_dados()` carrega os dados de um arquivo CSV (substitua `'dados_amazon.csv'` pelo caminho do seu arquivo).

### 2. **Previsão de Demanda**
A função `prever_demanda(dados)` treina um modelo de **Regressão Linear** com os dados de preço, estoque, descontos e sazonalidade para prever a demanda de produtos. O modelo é avaliado utilizando o **erro absoluto médio** e gera um gráfico comparando as previsões com os valores reais.

### 3. **Otimização de Fornecimento**
A função `otimizar_fornecimento(previsoes)` utiliza **Programação Linear** (via a função `linprog` da biblioteca `scipy`) para otimizar o fornecimento de produtos, minimizando os custos dentro dos limites de estoque disponíveis.

### 4. **Execução Principal**
A função `main()` executa todo o processo: carrega os dados, realiza a previsão de demanda e otimiza o fornecimento.

## Como Executar

1. Instale as bibliotecas necessárias com o comando:
   ```bash
   pip install pandas scikit-learn scipy matplotlib seaborn
2. Execute o script Python:
   ```bash
   python suply_chain_forecast.py
