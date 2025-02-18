import numpy as np
import pandas as pd
from scipy.stats import kstest, expon, norm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

class CrashPredictor:
    def __init__(self, data):
        """
        Inicializa o predictor com dados históricos de crash.
        :param data: Lista de pontos de crash (floats)
        """
        self.data = np.array(data)
        self.model = None

    def analyze_distribution(self):
        """
        Analisa a distribuição dos dados históricos.
        """
        print("Análise de Distribuição:")
        
        # Teste de Kolmogorov-Smirnov para verificar se os dados seguem uma distribuição exponencial
        exp_lambda = 1 / np.mean(self.data)  # Parâmetro lambda para distribuição exponencial
        ks_stat, p_value = kstest(self.data, 'expon', args=(0, exp_lambda))
        print(f"Distribuição Exponencial (λ={exp_lambda:.2f}): KS Statistic={ks_stat:.4f}, P-value={p_value:.4f}")

        # Teste para distribuição normal
        ks_stat, p_value = kstest(self.data, 'norm', args=(np.mean(self.data), np.std(self.data)))
        print(f"Distribuição Normal: KS Statistic={ks_stat:.4f}, P-value={p_value:.4f}")

        if p_value > 0.05:
            print("Os dados podem seguir essa distribuição.")
        else:
            print("Os dados não seguem essa distribuição.")

    def detect_patterns(self):
        """
        Detecta padrões simples nos dados históricos.
        """
        print("\nDetecção de Padrões:")
        diffs = np.diff(self.data)
        avg_diff = np.mean(diffs)
        std_diff = np.std(diffs)

        print(f"Média das diferenças entre crashes: {avg_diff:.2f}")
        print(f"Desvio padrão das diferenças: {std_diff:.2f}")

        if abs(avg_diff) < 0.1 * np.mean(self.data):
            print("Possível padrão: Diferenças pequenas entre crashes consecutivos.")
        else:
            print("Nenhum padrão óbvio detectado nas diferenças.")

    def train_model(self):
        """
        Treina um modelo de regressão para prever o próximo crash.
        """
        print("\nTreinando Modelo de Previsão...")
        X = np.arange(len(self.data)).reshape(-1, 1)  # Índices como features
        y = self.data  # Pontos de crash como target

        # Dividir dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Treinar modelo Random Forest
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Avaliar o modelo
        predictions = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        print(f"Erro Médio Absoluto no conjunto de teste: {mae:.2f}")

    def predict_next_crash(self):
        """
        Faz uma previsão para o próximo crash.
        """
        if self.model is None:
            raise ValueError("Modelo não treinado. Chame train_model() primeiro.")

        next_index = len(self.data)
        prediction = self.model.predict([[next_index]])[0]
        print(f"\nPrevisão para o próximo crash: {prediction:.2f}")
        return prediction

    def simulate_strategy(self, stop_loss=2.0, take_profit=1.5):
        """
        Simula uma estratégia de apostas com base nas previsões.
        :param stop_loss: Limite de perda antes de sair.
        :param take_profit: Limite de lucro antes de sair.
        """
        print("\nSimulação de Estratégia:")
        balance = 100.0  # Saldo inicial
        history = []

        for i in range(len(self.data)):
            current_crash = self.data[i]
            predicted_crash = self.predict_next_crash()

            # Decisão de quando sair
            if predicted_crash >= take_profit:
                outcome = min(current_crash, take_profit)  # Sair no take_profit
                profit = outcome - 1  # Lucro líquido
            else:
                outcome = 0  # Perde tudo se o crash ocorrer antes do take_profit
                profit = -1  # Perda total

            balance += profit
            history.append((current_crash, predicted_crash, profit, balance))

            print(f"Rodada {i+1}: Real={current_crash:.2f}, Previsto={predicted_crash:.2f}, Lucro={profit:.2f}, Saldo={balance:.2f}")

        df = pd.DataFrame(history, columns=["Real", "Previsto", "Lucro", "Saldo"])
        print("\nResumo da Simulação:")
        print(df.tail())

        print(f"\nSaldo Final: {balance:.2f}")


# Exemplo de uso
if __name__ == "__main__":
    # Dados históricos de crash (substitua pelos seus próprios dados)
    historical_data = [
        1.5, 2.3, 3.7, 1.8, 4.2, 2.9, 1.6, 5.1, 2.4, 3.0,
        1.9, 2.7, 3.5, 1.7, 4.0, 2.8, 1.5, 5.3, 2.5, 3.1,
        2.0, 2.6, 3.6, 1.8, 4.1, 2.9, 1.6, 5.2, 2.4, 3.0
    ]

    predictor = CrashPredictor(historical_data)

    # Etapa 1: Analisar distribuição
    predictor.analyze_distribution()

    # Etapa 2: Detectar padrões
    predictor.detect_patterns()

    # Etapa 3: Treinar modelo de previsão
    predictor.train_model()

    # Etapa 4: Prever próximo crash
    predictor.predict_next_crash()

    # Etapa 5: Simular estratégia de apostas
    predictor.simulate_strategy(stop_loss=2.0, take_profit=1.5)
