# Detecção Automatizada de Falhas em Painéis Solares utilizando YOLO26n

Este repositório contém o código-fonte desenvolvido para o Trabalho de Conclusão de Curso (TCC) que será apresentado à **Universidade Federal de Pernambuco (UFPE)** no ano de 2026. 

O projeto consiste no desenvolvimento e treinamento experimental de um modelo de deep learning baseado na arquitetura **YOLO26n** para a identificação e classificação automatizada de anomalias em módulos fotovoltaicos a partir de imagens multiespectrais.

## 📌 Visão Geral do Projeto

A inspeção manual de painéis solares é um processo demorado e sujeito a falhas. Este script implementa um pipeline completo para automatizar essa tarefa via visão computacional, executado em ambiente de nuvem (Google Colab) com persistência de dados e resultados estruturada diretamente no Google Drive.

O fluxo de execução compreende:
1. Configuração do ambiente e validação de hardware (GPU Tesla T4).
2. Download automatizado e estruturação do banco de dados multiespectral via Roboflow API.
3. Treinamento parametrizado do modelo preditivo com mecanismos de parada antecipada (*Early Stopping*).
4. Validação rigorosa em ambiente cego (partição de teste).
5. Geração de inferências visuais com delimitação de caixas envolventes (*bounding boxes*).

## 🛠️ Tecnologias e Frameworks

* **Linguagem:** Python 3
* **Framework Principal:** [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (Ambiente configurado para YOLO26n)
* **Gestão de Dataset:** [Roboflow SDK](https://roboflow.com/)
* **Ambiente de Execução:** Google Colab (utilizando aceleração por GPU)
* **Persistência de Dados:** Google Drive API

## 📋 Configuração do Hiperparâmetros de Treinamento

O modelo foi treinado sob diretrizes rigorosas para garantir a convergência e evitar o *overfitting*, utilizando os seguintes parâmetros estruturais:

| Hiperparâmetro | Valor Estipulado | Motivação / Papel no Pipeline |
| :--- | :--- | :--- |
| **Epochs** | 150 | Teto máximo de iterações para a convergência dos pesos. |
| **Patience** | 20 | Early stopping para interromper o treino se não houver evolução. |
| **Imgsz** | 640 | Resolução matricial otimizada para detecção de pequenas falhas. |
| **Batch** | 16 | Dimensão do lote adequada à restrição de VRAM da GPU. |
| **Seed** | 42 | Semente determinística adotada para reprodutibilidade dos testes. |
| **Optimizer** | AdamW | Otimizador com decaimento adaptativo de peso. |

## 🚀 Como Executar

### Pré-requisitos

Antes de rodar o script, certifique-se de possuir:
1. Uma conta no **Roboflow** com acesso ao dataset do projeto e sua chave privada de API (`api_key`).
2. Espaço disponível no **Google Drive** para a criação da estrutura de pastas `/TCC_Solar/`.

### Passo a Passo

1.  **Clone este repositório:**
    ```bash
    git clone [https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO.git](https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO.git)
    ```
2.  Importe o arquivo do script ou notebook para o seu ambiente do **Google Colab**.
3.  No bloco de código correspondente à inicialização da API do Roboflow, substitua o comentário pelo seu token privado:
    ```python
    rf = Roboflow(api_key="SUA_CHAVE_PRIVADA_AQUI")
    ```
4.  Execute as células sequencialmente. Certifique-se de conceder a permissão de leitura/escrita para o Google Drive quando a janela pop-up de autenticação for exibida.

## 📊 Métricas de Avaliação

O script está configurado para isolar a partição de testes ao final do processo e extrair as seguintes métricas oficiais de desempenho:
* **mAP@50** (Mean Average Precision)
* **Precisão** (Precision)
* **Revocação** (Recall)

Os melhores pesos gerados durante as rodadas (`best.pt`) e as imagens resultantes das predições serão salvos automaticamente no diretório mapeado do seu Drive para posterior inclusão na documentação textual da pesquisa.

## ✒️ Autor

* **Mateus Roberto Alves da Silva** - *Desenvolvimento e Pesquisa* - UFPE (2026)
