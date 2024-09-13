# Importação das bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import seaborn as sns

# Função para carregar e limpar os dados
def carregar_dados():
    # Carregar dados de um CSV (substitua 'dados_amazon.csv' pelo seu arquivo)
    dados = pd.read_csv('dados_amazon.csv')
    return dados

# Função para treinar o modelo de Machine Learning e prever a demanda
def prever_demanda(dados):
    # Selecionar features e target
    X = dados[['preço', 'estoque', 'descontos', 'sazonalidade']]
    y = dados['demanda']

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar o modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Fazer previsões
    previsoes = modelo.predict(X_test)

    # Avaliar o modelo
    erro = mean_absolute_error(y_test, previsoes)
    print(f"Erro absoluto médio: {erro}")

    # Gráfico comparando previsões e valores reais
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=y_test, y=previsoes, color="blue", label="Previsões")
    sns.lineplot(x=y_test, y=y_test, color="red", label="Real")
    plt.title("Comparação entre valores reais e previstos")
    plt.xlabel("Valores Reais de Demanda")
    plt.ylabel("Previsões de Demanda")
    plt.legend()
    plt.show()

    # Retornar as previsões (para otimização posterior)
    return previsoes[:3]  # Exemplo, pegar previsões para 3 produtos

# Função para otimizar o fornecimento com base nas previsões de demanda
def otimizar_fornecimento(previsoes):
    # Exemplo de custos para cada produto
    custos = [20, 30, 50]

    # Limite de estoque (exemplo)
    limite_estoque = [120, 220, 180]

    # Definir as restrições (limite de estoque)
    A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    b = limite_estoque

    # Executar a otimização
    res = linprog(custos, A_ub=A, b_ub=b, bounds=(0, None))

    print("Quantidade de produtos a ser fornecida para minimizar o custo:", res.x)

# Função principal que executa todo o processo
def main():
    dados = carregar_dados()
    previsoes = prever_demanda(dados)
    otimizar_fornecimento(previsoes)

# Executar o script
if __name__ == "__main__":
    main()
