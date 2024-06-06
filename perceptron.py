import numpy as np

class Perceptron:
    def __init__(self):
        pass

    #Usado após criar a rede para treinar
    #modelo de IA -> Aprendizado
    def treinamento(self, entradas, saidas, taxaAprendizado, epocas):
        self.entradas = entradas
        self.saidas = saidas
        self.taxaAprendizado = taxaAprendizado
        self.epocas = epocas

        #Pesos e bias gerados aletoriamente
        #na primeira etapa do treinamento
        w1 = np.random.uniform(-1,1)
        w2 = np.random.uniform(-1,1)
        bias = np.random.uniform(-1,1)

        for _ in range(self.epocas):
            for i in range(len(self.entradas)):
                #Computar a saída prevista
                #Aplicar uma função de ativação: SIGMOID
                x = (self.entradas[i][0] * w1) + (self.entradas[i][1] * w2) + bias
                sigmoid = self.sigmoid(x)

                w1 += self.taxaAprendizado * (self.saidas[i][0] - sigmoid) * self.entradas[i][0]
                w2 += self.taxaAprendizado * (self.saidas[i][0] - sigmoid) * self.entradas[i][1]

                bias += self.taxaAprendizado * (self.saidas[i][0] - sigmoid)

        self.w1 = w1
        self.w2 = w2
        self.bias = bias

    #Função de ativação
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #Após treinar ela é usada para predizer
    #a classe de uma Entrada qualquer
    def predicao(self, entrada):
        x = np.dot(entrada[0], self.w1) + np.dot(entrada[1], self.w2) + self.bias
        return self.sigmoid(x)