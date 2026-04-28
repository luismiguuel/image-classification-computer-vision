# 🛣️ Classificação de Pavimentos com PyTorch (Transfer Learning)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

Este repositório contém a solução para um desafio técnico de Visão Computacional. O objetivo é classificar imagens de diferentes tipos de pavimentos em três categorias: **Asphalt** (Asfalto), **Belgian Blocks** (Blocos Belgas) e **Off-road** (Terra/Não pavimentado).

## 🧠 Sobre o Projeto e Desafios

O projeto lida com um cenário altamente realista e desafiador no mundo da Ciência de Dados:
* **Dataset Reduzido:** Poucas imagens disponíveis para treino (apenas 900 imagens).
* **Desbalanceamento Severo:** A classe `Asphalt` domina o conjunto de dados, enquanto `Belgian Blocks` é uma classe extremamente minoritária.
* **Condições Adversas:** Imagens capturadas em diferentes condições de iluminação, incluindo período noturno.

Para contornar o risco de *overfitting* e a escassez de dados, a solução foi construída utilizando **Transfer Learning** a partir da arquitetura **ResNet18** pré-treinada no ImageNet.

## 🔬 Abordagem e Experimentos

O projeto foi desenvolvido de forma iterativa, aplicando o método científico para testar hipóteses sobre o comportamento da rede neural face aos dados desbalanceados.

### 1. Baseline (Solução Inicial)
* **Configuração:** ResNet18 com a última camada ajustada para 3 classes, treinada por 5 épocas com `CrossEntropyLoss` padrão.
* **Resultado:** Acurácia global de 87%.
* **Análise Crítica:** A métrica de acurácia revelou-se ilusória. O modelo adotou um comportamento enviesado, alcançando 94% de *recall* na classe majoritária, mas apenas **25% de recall** nos Blocos Belgas (a rede simplesmente ignorava a classe minoritária).

### 2. Experimento 1: Pesos por Classe (Solução Ideal)
* **Hipótese:** Penalizar severamente os erros cometidos na classe minoritária forçará o modelo a aprender os seus padrões.
* **O que foi feito:** Cálculo estatístico de pesos (inversamente proporcionais à frequência das classes) via `scikit-learn` e aplicação destes no critério de perda (`CrossEntropyLoss`).
* **Resultado:** O *recall* da classe `Belgian Blocks` **saltou de 25% para 62%**, com o *F1-Score macro* atingindo o seu pico (0.83). O modelo tornou-se muito mais equilibrado.

### 3. Experimento 2: Data Augmentation + Pesos
* **Hipótese:** Aplicar rotações, espelhamentos e variações de cor simulará as condições visuais adversas e melhorará a generalização.
* **O que foi feito:** Adição de `RandomHorizontalFlip`, `RandomRotation` e `ColorJitter` ao pipeline de treino.
* **Resultado e Trade-off:** A acurácia global atingiu o máximo (90.67%), mas o *recall* dos Blocos Belgas caiu para 44%. 
* **Análise Crítica:** Como os pavimentos de blocos dependem de um padrão geométrico muito rígido, a aplicação de transformações aleatórias em poucas amostras acabou por descaracterizar a textura geométrica da classe, prejudicando o seu reconhecimento.

## 🛠️ Tecnologias Utilizadas

* **Python 3**
* **PyTorch & Torchvision:** Construção de Datasets, DataLoaders, Transfer Learning e Data Augmentation.
* **Scikit-Learn:** Cálculo de métricas de avaliação (`classification_report`, `confusion_matrix`) e pesos de balanceamento.
* **Matplotlib & Seaborn:** Visualização de dados e plotagem das Matrizes de Confusão.
* **Jupyter Notebook:** Ambiente de desenvolvimento e documentação técnica.

## 🚀 Como Executar

1. Clone este repositório:
   ```bash
   git clone https://github.com/luismiguuel/image-classification-computer-vision.git

2. Instale as dependências necessárias:
    ```bash
    pip install torch torchvision scikit-learn matplotlib seaborn jupyter

3. Certifique-se de que a pasta com as imagens está na raiz do projeto (estruturada em pastas de treino e teste por classe).

4. Inicie o Jupyter Notebook:
    ```bash
    jupyter notebook

5. Abra o ficheiro `solucao.ipynb` e execute as células sequencialmente (Run All). O código está preparado para utilizar aceleração via GPU (CUDA) automaticamente, caso esteja disponível.

## Conclusão
A análise demonstrou que, em datasets desbalanceados onde a classe minoritária depende de padrões geométricos estruturados, a alteração da Função de Perda (Pesos por Classe) é significativamente mais eficaz e segura do que a aplicação agressiva de Data Augmentation.