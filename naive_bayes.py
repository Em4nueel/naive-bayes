import numpy as np                       # importa a biblioteca numpy com alias np para operações numéricas
from sklearn.naive_bayes import GaussianNB  # importa o classificador Gaussian Naive Bayes do scikit-learn
from sklearn.model_selection import train_test_split  # importa função para dividir dados em treino/teste (não usada aqui, mas mantida)
from sklearn.metrics import accuracy_score  # importa função para calcular acurácia do modelo
import matplotlib.pyplot as plt           # importa matplotlib.pyplot com alias plt para visualização

# 1. Gerar um dataset simples
# Vamos criar 20 amostras com 2 características e 2 classes
np.random.seed(0) # Para reprodutibilidade das amostras aleatórias (fixa a semente do gerador RNG)
X = np.concatenate([                      # concatena os blocos de amostras geradas para compor X (matriz de características)
    np.random.randn(10, 2) + np.array([0, 0]), # gera 10 amostras (10x2) para a Classe 0 centradas em (0,0)
    np.random.randn(10, 2) + np.array([5, 5])  # gera 10 amostras (10x2) para a Classe 1 centradas em (5,5)
])
y = np.concatenate([                      # concatena os vetores de rótulos para compor y (vetor de classes)
    np.zeros(10), # 10 amostras da classe 0 (rótulo 0)
    np.ones(10)   # 10 amostras da classe 1 (rótulo 1)
])

# 2. Visualizar o dataset (opcional, mas ajuda a entender)
plt.figure(figsize=(7, 5))                # cria uma nova figura com tamanho definido (7x5 polegadas)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='Classe 0')  # plota pontos da Classe 0 em vermelho
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Classe 1') # plota pontos da Classe 1 em azul
plt.title('Dataset Simples para Naive Bayes') # define o título do gráfico
plt.xlabel('Característica 1')             # rotula o eixo X
plt.ylabel('Característica 2')             # rotula o eixo Y
plt.legend()                               # exibe a legenda com os rótulos das classes
plt.grid(True)                             # exibe a grade no gráfico para melhor leitura
plt.show()                                 # mostra o gráfico na tela

# 3. Dividir os dados em treino e teste (geralmente fazemos isso, mas para um exemplo pequeno podemos usar todos os dados)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # exemplo comentado de split

# 4. Criar e Treinar o Modelo Naive Bayes
modelo_simples = GaussianNB() # instancia o classificador Gaussian Naive Bayes
modelo_simples.fit(X, y) # Treinando com todos os dados para simplificar  # treina o modelo usando todas as amostras X e rótulos y

# 5. Fazer Previsões
previsoes = modelo_simples.predict(X) # prevê as classes para as mesmas amostras usadas no treino (apenas para exemplo)

# 6. Avaliar o Modelo
acuracia_simples = accuracy_score(y, previsoes)  # calcula a acurácia comparando rótulos verdadeiros e previsões
print(f"\nAcurácia do modelo Gaussian Naive Bayes: {acuracia_simples:.2f}")  # imprime a acurácia formatada com 2 casas decimais

# Exemplo de previsão para um novo ponto
novo_ponto_1 = np.array([[1, 1]]) # Mais próximo da Classe 0  # cria um novo ponto (1,1) no formato 2D esperado pelo predict
previsao_1 = modelo_simples.predict(novo_ponto_1)  # prevê a classe do novo ponto
print(f"Previsão para o ponto [1, 1]: Classe {int(previsao_1[0])}")  # imprime a previsão convertida para inteiro

novo_ponto_2 = np.array([[4, 6]]) # Mais próximo da Classe 1  # cria outro ponto (4,6) para testar previsão
previsao_2 = modelo_simples.predict(novo_ponto_2)  # prevê a classe do segundo novo ponto
print(f"Previsão para o ponto [4, 6]: Classe {int(previsao_2[0])}")  # imprime a previsão do segundo ponto
