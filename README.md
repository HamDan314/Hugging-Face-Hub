# Prototipo PLN para comparación de modelos de análisis de sentimientos

## Descripción del proyecto

Este repositorio presenta un prototipo de Procesamiento de Lenguaje Natural para clasificar automáticamente el sentimiento de frases y reseñas breves en inglés. Su propósito es comparar tres modelos preentrenados de Hugging Face sobre el dataset SST-2, utilizando métricas comunes y el tiempo promedio de inferencia.

La comparación permite identificar qué alternativa ofrece un equilibrio adecuado entre calidad de las predicciones y velocidad de respuesta. Se utilizan los modelos disponibles sin realizar entrenamiento desde cero ni ajuste fino.

Los tres modelos evaluados son:

| Modelo | Identificador en Hugging Face | Propósito en la comparación |
|---|---|---|
| DistilBERT SST-2 | `distilbert-base-uncased-finetuned-sst-2-english` | Evaluar una alternativa ligera y rápida. |
| BERT SST-2 | `textattack/bert-base-uncased-SST-2` | Analizar el equilibrio entre desempeño y eficiencia. |
| RoBERTa Sentiment | `siebert/sentiment-roberta-large-english` | Valorar una alternativa de alto desempeño y mayor demanda computacional. |

## Ecosistema Hugging Face utilizado

El prototipo integra los siguientes componentes:

- **transformers:** carga de los modelos y creación del proceso de inferencia.
- **datasets:** obtención del dataset y selección reproducible de la muestra.
- **huggingface_hub:** acceso y autenticación opcional en el Hub.
- **Model Hub:** distribución de los modelos preentrenados utilizados.
- **pipeline:** interfaz para ejecutar la clasificación de sentimientos.

Los modelos y datos públicos pueden utilizarse sin autenticación. El notebook conserva la opción de iniciar sesión mediante `notebook_login()`; los tokens personales no deben guardarse en el repositorio.

## Dataset

Se utiliza **SST-2 (Stanford Sentiment Treebank v2)**, parte del benchmark GLUE y disponible mediante Hugging Face Datasets.

```python
from datasets import load_dataset

dataset = load_dataset("nyu-mll/glue", "sst2")
```

Sus campos principales son:

| Campo | Descripción |
|---|---|
| `sentence` | Frase en inglés que será clasificada. |
| `label` | Clase de sentimiento: 0 = negativo y 1 = positivo. |
| `idx` | Identificador del registro. |

La evaluación emplea **300 registros de validación**, seleccionados con `shuffle(seed=42)`. Todos los modelos se comparan sobre la misma muestra, lo que facilita una evaluación consistente.

```python
eval_data = dataset["validation"].shuffle(seed=42).select(range(300))
```

El dataset se descarga mediante la biblioteca; no es necesario incorporar todos sus archivos al repositorio. La carpeta `data` contiene la descripción de su origen y estructura.

## Estructura del repositorio

El archivo `prototipo_pln_sentimientos.zip` contiene el proyecto organizado de la siguiente manera:

```text
prototipo_pln_sentimientos/
├── README.md
├── requirements.txt
├── data/
│   └── dataset_info.md
├── notebooks/
│   └── cuaderno_colab_pln_sentimientos.ipynb
├── results/
│   └── resultados_modelos_pln.csv
└── src/
    └── evaluate_models.py
```

El README y las dependencias también se incluyen en la raíz del repositorio para facilitar su consulta. Para ejecutar el proyecto localmente, se debe descomprimir el paquete y trabajar desde su carpeta principal.

## Entorno de ejecución

El cuaderno está preparado para utilizarse en **Google Colab**, con opción de aceleración mediante GPU. También permite inferencia en CPU cuando no se dispone de GPU.

Configuración recomendada:

- Python 3.10 o superior.
- Conexión a internet para descargar modelos y dataset.
- Dependencias indicadas en `requirements.txt`.
- GPU opcional para acelerar la inferencia.

En Colab, la GPU se selecciona desde **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU**. Su disponibilidad puede comprobarse con:

```python
import torch

print("GPU disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

Instalación de dependencias:

```bash
pip install -r requirements.txt
```

La latencia depende del hardware, las versiones de las bibliotecas y las condiciones de ejecución. Por ello, los tiempos registrados deben interpretarse dentro del entorno en el que se obtuvieron.

## Pasos de ejecución

### Opción 1 Ejecutar el cuaderno en Colab

1. Abrir `notebooks/cuaderno_colab_pln_sentimientos.ipynb` en Google Colab.
2. Seleccionar GPU si está disponible.
3. Ejecutar las celdas en orden para instalar dependencias y comprobar el entorno.
4. Utilizar la autenticación opcional al Hub cuando se requiera.
5. Cargar el dataset y la muestra de evaluación.
6. Ejecutar la inferencia de los tres modelos.
7. Revisar la tabla comparativa y los reportes por modelo.
8. Exportar las métricas en formato CSV.

Cuaderno compartido:

https://colab.research.google.com/drive/168ZqhO9nofndRaBh12DNOgz-azOkEf_E?usp=sharing

### Opción 2 Ejecutar el script

Desde la raíz del proyecto descomprimido:

```bash
python src/evaluate_models.py
```

El script obtiene los datos, evalúa los modelos y guarda las métricas en:

```text
results/resultados_modelos_pln.csv
```

Una nueva ejecución puede producir tiempos diferentes a los registrados originalmente, especialmente si se cambia entre CPU y GPU.

## Métricas utilizadas

Se consideran cinco indicadores:

- **Accuracy:** proporción de predicciones correctas respecto al total evaluado.
- **Precision:** proporción de opiniones realmente positivas entre las clasificadas como positivas.
- **Recall:** proporción de opiniones positivas reales que el modelo identifica.
- **F1-score:** media armónica de precision y recall; resume su equilibrio.
- **Latencia:** tiempo promedio de inferencia por texto, expresado en milisegundos.

Los reportes de clasificación y las matrices de confusión complementan la tabla general y permiten identificar errores por clase. Las métricas de precision, recall y F1 presentadas en la tabla corresponden a la clase positiva.

## Resultados principales

Los valores conservados corresponden a la ejecución original documentada sobre la muestra de 300 textos.

| Modelo | Accuracy | Precision | Recall | F1-score | Latencia ms por texto |
|---|---:|---:|---:|---:|---:|
| DistilBERT SST-2 | 0.9133 | 0.9018 | 0.9363 | 0.9188 | 63.39 |
| BERT SST-2 | 0.9233 | 0.9351 | 0.9172 | 0.9260 | 136.19 |
| RoBERTa Sentiment | 0.9233 | 0.9136 | 0.9427 | 0.9279 | 495.16 |

BERT y RoBERTa alcanzan la misma accuracy. BERT obtiene la mayor precision, mientras que RoBERTa presenta el mayor recall y F1-score. DistilBERT responde en menos tiempo que las otras alternativas.

## Conclusiones

**DistilBERT** destaca por su eficiencia: registra 63.39 ms por texto y un F1-score de 0.9188. Resulta una opción adecuada cuando la velocidad y la disponibilidad de recursos son criterios prioritarios.

**RoBERTa** logra el mayor F1-score, de 0.9279, y el mayor recall, de 0.9427. Su latencia de 495.16 ms por texto exige valorar si la mejora de desempeño compensa el mayor tiempo de respuesta.

**BERT SST-2** ofrece un equilibrio favorable: alcanza accuracy de 0.9233 y F1-score de 0.9260, con latencia de 136.19 ms por texto. Por esta relación entre desempeño y eficiencia, se recomienda como modelo principal del prototipo.

## Justificación final

La recomendación de BERT considera conjuntamente la confiabilidad de las predicciones y la velocidad de inferencia. La diferencia de F1-score respecto a RoBERTa es pequeña en esta muestra, mientras que su tiempo de respuesta es menor. DistilBERT se mantiene como alternativa cuando se necesita priorizar rapidez.

Esta selección constituye una conclusión del prototipo y no una garantía de desempeño en producción. Antes de desplegarlo, se deben evaluar textos del contexto real, revisar errores y medir consumo de recursos bajo la carga prevista. No se calcularon costos monetarios de operación.

## Alcance y riesgos

La comparación se limita a frases en inglés y dos categorías de sentimiento. No valida el desempeño sobre español, opiniones neutrales, sarcasmo o textos de otros sectores. Una aplicación real requiere revisar sesgos, proteger información personal y definir cuándo una predicción debe pasar a revisión humana.

No se documenta un ajuste específico de umbrales ni una calibración de probabilidades. Estos elementos deben validarse antes de utilizar los resultados para decisiones operativas.

## Actualización de documentación

Esta versión reformula y amplía la explicación del proyecto, manteniendo los modelos, el dataset, el código y las métricas de la ejecución original. La actualización de la documentación no representa una nueva ejecución experimental.

Repositorio del proyecto:

https://github.com/HamDan314/DAn-Alan-Cardoso
