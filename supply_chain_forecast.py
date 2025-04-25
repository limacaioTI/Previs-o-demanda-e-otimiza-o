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
    dados = pd.read_csv('dados_amazon.csv')
    return dados

# Função para treinar o modelo de Machine Learning e prever a demanda
def prever_demanda(dados):
    X = dados[['preço', 'estoque', 'descontos', 'sazonalidade']]
    y = dados['demanda']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    previsoes = modelo.predict(X_test)

    erro = mean_absolute_error(y_test, previsoes)
    print(f"Erro absoluto médio: {erro:.2f}")

    plt.figure(figsize=(10,6))
    sns.scatterplot(x=y_test, y=previsoes, color="blue", label="Previsões")
    sns.lineplot(x=y_test, y=y_test, color="red", label="Real")
    plt.title("Comparação entre valores reais e previstos")
    plt.xlabel("Valores Reais de Demanda")
    plt.ylabel("Previsões de Demanda")
    plt.legend()
    plt.show()

    return previsoes[:3]

# Função para otimizar o fornecimento com base nas previsões de demanda
def otimizar_fornecimento(previsoes):
    custos = [20, 30, 50]
    limite_estoque = [120, 220, 180]

    # Reduzimos levemente as demandas previstas para garantir viabilidade
    demanda_minima = [min(p, l) for p, l in zip(previsoes, limite_estoque)]

    A_eq = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    b_eq = demanda_minima

    A_ub = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    b_ub = limite_estoque

    res = linprog(c=custos, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=(0, None))

    if res.success:
        print("\nQuantidade de produtos a ser fornecida para minimizar o custo:")
        print(res.x)

        custo_total = sum([c * x for c, x in zip(custos, res.x)])
        print(f"Custo total estimado: R${custo_total:.2f}")
    else:
        print("\n⚠️ A otimização falhou:", res.message)

# Função principal que executa todo o processo
def main():
    dados = carregar_dados()
    previsoes = prever_demanda(dados)
    otimizar_fornecimento(previsoes)

# Executar o script
if __name__ == "__main__":
    main()
