import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

def main():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    print(" Carregando dataset...")
    df = pd.read_csv(url)
    
    # Cálculo objetivo da taxa de contaminação (Fraudes / Total)
    taxa_contaminacao = df['Class'].sum() / len(df)
    
    # Separação de features e target
    X = df.drop('Class', axis=1)
    y_gabarito = df['Class']
    
    print(" Treinando Isolation Forest...")
    modelo = IsolationForest(
        contamination=taxa_contaminacao,
        random_state=42,
        n_jobs=-1
    )
    
    # Predição: 1 para normal, -1 para outlier
    previsoes = modelo.fit_predict(X)
    
    # Mapeamento matemático: -1 vira 1 (Fraude) e 1 vira 0 (Normal)
    previsoes_binarias = np.where(previsoes == -1, 1, 0)
    
    # Métricas de performance
    print("\n Relatório de Classificação:")
    print(classification_report(y_gabarito, previsoes_binarias, target_names=['Normal (0)', 'Fraude (1)']))
    
    # Exportação da Matriz de Confusão
    plt.figure(figsize=(5, 4))
    cm = confusion_matrix(y_gabarito, previsoes_binarias)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Previsto Normal', 'Previsto Fraude'],
                yticklabels=['Real Normal', 'Real Fraude'])
    plt.title('Matriz de Confusão')
    plt.tight_layout()
    plt.savefig('matriz_confusao.png', dpi=300)
    print("💾 Gráfico 'matriz_confusao.png' salvo.")

if __name__ == "__main__":
    main()
