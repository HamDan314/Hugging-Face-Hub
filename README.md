# Evaluación de modelos para análisis de sentimientos

Este proyecto compara tres modelos preentrenados de Hugging Face para clasificar frases en inglés como positivas o negativas. La evaluación utiliza 300 registros de validación de SST-2, seleccionados con la semilla 42.

## Modelos evaluados

- DistilBERT: `distilbert-base-uncased-finetuned-sst-2-english`
- BERT: `textattack/bert-base-uncased-SST-2`
- RoBERTa: `siebert/sentiment-roberta-large-english`

## Resultados registrados

| Modelo | Accuracy | Precision | Recall | F1-score | Latencia ms por texto |
|---|---:|---:|---:|---:|---:|
| DistilBERT | 0.9133 | 0.9018 | 0.9363 | 0.9188 | 63.39 |
| BERT | 0.9233 | 0.9351 | 0.9172 | 0.9260 | 136.19 |
| RoBERTa | 0.9233 | 0.9136 | 0.9427 | 0.9279 | 495.16 |

BERT se recomienda por su equilibrio entre desempeño y velocidad. DistilBERT es la alternativa más rápida; RoBERTa obtiene el mayor F1-score y recall, con mayor latencia. Estos valores corresponden a la ejecución documentada y no a una nueva evaluación. La latencia depende del hardware y del entorno.

## Dataset

SST-2 pertenece al benchmark GLUE y está disponible mediante Hugging Face Datasets. Los campos son `sentence` (frase), `label` (0 negativo y 1 positivo) e `idx` (identificador).

## Contenido

- `notebooks/cuaderno_colab_pln_sentimientos.ipynb`: cuaderno de evaluación.
- `src/evaluate_models.py`: script reproducible.
- `results/resultados_modelos_pln.csv`: métricas registradas.
- `data/dataset_info.md`: descripción del dataset.
- `requirements.txt`: dependencias.

## Ejecución

Instala las dependencias con `pip install -r requirements.txt`. Ejecuta el notebook en Google Colab o utiliza `python src/evaluate_models.py` desde la raíz del proyecto. La GPU es opcional para la inferencia; la autenticación al Hub es opcional para estos modelos públicos.

Cuaderno: https://colab.research.google.com/drive/168ZqhO9nofndRaBh12DNOgz-azOkEf_E?usp=sharing

Repositorio: https://github.com/HamDan314/DAn-Alan-Cardoso

## Alcance y limitaciones

La evaluación se limita a frases en inglés y clasificación binaria. No demuestra desempeño sobre español, sarcasmo u opiniones neutrales. Antes de una implementación real se requiere validación con datos del contexto objetivo y revisión de errores.

## Actualización de documentación

Se reformularon la descripción, la interpretación de resultados y las instrucciones de uso. Se conservan el código y las métricas de la ejecución original.
