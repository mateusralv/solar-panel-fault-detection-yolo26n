# ==============================================================================
# UNIVERSIDADE FEDERAL DE PERNAMBUCO (UFPE)
# SCRIPT DE TREINAMENTO EXPERIMENTAL - YOLO26n
# AUTOR: MATEUS ROBERTO ALVES DA SILVA
# ANO: 2026
# ==============================================================================

import os
import torch
from roboflow import Roboflow
from ultralytics import YOLO

# NOTA: Em um ambiente Colab, a montagem do drive é necessária. 
# Se for rodar localmente, comente as duas linhas abaixo e ajuste os caminhos 'project'.
from google.colab import drive
drive.mount('/content/drive')

print(f"GPU disponível para treinamento: {torch.cuda.is_available()}")

# --------------------------------------------------------------------------
# 2. Autenticação e Download do Dataset Multiespectral (Roboflow API)
# --------------------------------------------------------------------------
# Boa prática: Usar variáveis de ambiente para esconder a chave de API
api_key = os.environ.get("ROBOFLOW_API_KEY", "SUA_CHAVE_AQUI")

rf = Roboflow(api_key=api_key)
project_rf = rf.workspace("tcc-qifp7").project("solar-panel-faulty-dataset-final-trial-iitcq")
version = project_rf.version(1)

# Download do banco de dados na estrutura de pastas exigida pelo Ultralytics
dataset = version.download("yolov11")
yaml_path = f"{dataset.location}/data.yaml"

# --------------------------------------------------------------------------
# 3. Inicialização e Treinamento do Modelo YOLO26n
# --------------------------------------------------------------------------
# Carregamento da arquitetura de rede Nano com pesos pré-treinados (COCO)
model = YOLO('yolo26n.pt')

# Execução do treinamento parametrizado rigorosamente conforme Metodologia
results = model.train(
    data=yaml_path,
    epochs=150,                 # Teto de iterações para convergência
    patience=20,                # Early stopping para prevenir overfitting
    imgsz=640,                  # Resolução matricial otimizada
    batch=16,                   # Tamanho do lote para a VRAM da GPU
    seed=42,                    # Semente determinística para reprodutibilidade
    optimizer="AdamW",          # Otimizador com decaimento adaptativo
    project='/content/drive/MyDrive/TCC_Solar/treinamento_resultados',
    name='experimento_roboflow_v26'
)

# --------------------------------------------------------------------------
# 4. Validação e Extração de Métricas Oficiais (Teste Isolado)
# --------------------------------------------------------------------------
# Carregamento dos melhores pesos sinápticos consolidados pelo Early Stopping
best_model_path = '/content/drive/MyDrive/TCC_Solar/treinamento_resultados/experimento_roboflow_v26/weights/best.pt'
best_model = YOLO(best_model_path)

# Execução da validação restrita à partição de teste (ambiente cego)
metrics = best_model.val(data=yaml_path, split='test')

# Impressão das métricas primárias estruturais
print(f"mAP@50: {metrics.box.map50:.4f}")
print(f"Precisão (Precision): {metrics.box.mp:.4f}")
print(f"Revocação (Recall): {metrics.box.mr:.4f}")

# --------------------------------------------------------------------------
# 5. Inferência Visual e Pós-Processamento
# --------------------------------------------------------------------------
# Geração de inferências visuais com bounding boxes sobre as imagens isoladas
predict_results = best_model.predict(
    source=f"{dataset.location}/test/images",
    save=True,                  # Salva as imagens mapeadas no Google Drive
    conf=0.5,                   # Limiar de confiança mínimo de 50% estipulado
    project='/content/drive/MyDrive/TCC_Solar/teste_final'
)
