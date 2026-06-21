# deteccao-anomalias-transacoes
#  Detecção de Anomalias em Transações Financeiras

# 1. Visão Geral do Projeto
Este projeto tem como objetivo a construção de um pipeline de detecção de fraudes em cartões de crédito utilizando aprendizado de máquina não-supervisionado, especificamente o algoritmo **Isolation Forest**. O código foi desenvolvido como parte de um desafio técnico, focando na identificação de padrões anômalos em um conjunto de dados massivo e altamente desbalanceado.

## 2. O Conjunto de Dados (Dataset)
Os dados utilizados são oriundos do repositório oficial do TensorFlow (originalmente do Kaggle). 

* *Características (Features):* O dataset contém transações realizadas em um período de dois dias. Por questões de confidencialidade, as variáveis originais foram submetidas a uma Análise de Componentes Principais (PCA), resultando nas variáveis contínuas `V1` a `V28`. As únicas variáveis que não sofreram transformação são `Time` (segundos decorridos entre a transação e a primeira transação) e `Amount` (valor da transação).
* *Alvo (Target):* A variável `Class` assume o valor `1` em caso de fraude e `0` em transações normais.
* **Desbalanceamento:** Trata-se de um cenário crítico, onde as fraudes representam aproximadamente **0,17%** do total de transações.

## 3. Fundamentação Teórica: Isolation Forest
Para a modelagem matemática deste problema, optou-se pelo *Isolation Forest* (Floresta de Isolamento). Diferente de algoritmos de perfilamento (que tentam aprender o que é "normal"), este algoritmo isola as anomalias de forma explícita.

*Como funciona:*
O algoritmo constrói múltiplas árvores de decisão aleatórias (*random decision trees*). Ele seleciona uma variável aleatoriamente e faz cortes em valores aleatórios entre o máximo e o mínimo. Como as anomalias são, por definição, dados raros e com valores extremos, elas são isoladas em nós muito próximos à raiz da árvore (exigem menos cortes para serem separadas dos demais dados). O comprimento do caminho do nó raiz até o nó final define o "score" de anomalia.

## 4. Estrutura do Repositório


meu-projeto-deteccao-anomalias/
│
├── README.md               # Documentação principal do projeto
├── requirements.txt        # Dependências de bibliotecas
├── main.py                 # Pipeline de execução principal
└── notebooks/
    └── exploracao.ipynb    # Análise exploratória detalhada (EDA)


## 5. Resultados e Análise Crítica
Na execução padrão, o modelo de *Isolation Forest* atinge um excelente desempenho na identificação de transações normais (alta precisão). No entanto, enfrenta limitações intrínsecas ao lidar com fraudes altamente camufladas no espaço vetorial.

* **Limitações Observadas:** O *Recall* (capacidade de detectar todas as fraudes existentes) tende a ser baixo na configuração base. Isso ocorre porque o Isolation Forest tem dificuldade quando as anomalias não estão geometricamente distantes dos dados normais no espaço hiperdimensional formado pelo PCA.
* **Próximos Passos (Melhorias Futuras):** 1. Comparação de performance com o algoritmo **Local Outlier Factor (LOF)**, que trabalha com densidade local.
  2. Implementação de abordagens supervisionadas ou semi-supervisionadas (como *XGBoost* com SMOTE para balanceamento de classes).

## 6. Como Executar (Reproducibilidade)

**Pré-requisitos:**
Certifique-se de ter o Python 3.8+ instalado. Recomenda-se o uso de um ambiente virtual (venv ou conda).

**Instalação de Dependências:**
bash
pip install -r requirements.txt




**Execução do Pipeline:**
bash
python main.py

O script fará o download automático do dataset via URL do TensorFlow, executará o treinamento em memória e plotará a Matriz de Confusão para análise das métricas de erro.
