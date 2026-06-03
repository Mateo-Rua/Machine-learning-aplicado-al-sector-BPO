# 🤖 Machine Learning Aplicado al Sector BPO

## Optimización del Servicio al Cliente mediante Modelos Predictivos

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-green?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/LightGBM-Gradient%20Boosting-yellow" />
  <img src="https://img.shields.io/badge/Gemini_2.5_Flash-NLP-red?logo=google" />
  <img src="https://img.shields.io/badge/Status-Completo-brightgreen" />
</p>

---

## 📋 Descripción del Proyecto

Este proyecto aborda un problema real del sector **BPO (Business Process Outsourcing)**: rediseñar el esquema de atención al cliente de una línea de servicio hacia un modelo más personalizado y enfocado en las necesidades específicas de cada usuario.

Se desarrollan dos líneas de análisis complementarias: (1) modelos predictivos supervisados para clasificar el comportamiento del cliente a partir de su historial de consultas y perfil demográfico, y (2) análisis de lenguaje natural con LLM para identificar las causas raíz de insatisfacción a partir de los comentarios de las encuestas post-llamada.

---

## 🎯 Objetivos

1. **Perfilar al cliente** que se comunica a la línea de atención mediante análisis descriptivo, correlación y pruebas de hipótesis.
2. **Construir modelos predictivos** utilizando al menos 3 algoritmos supervisados para predecir la variable objetivo `y`.
3. **Comparar y seleccionar** el mejor modelo basado en métricas de rendimiento orientadas al contexto de negocio.
4. **Identificar motivos de insatisfacción** en los comentarios de encuestas mediante NLP y un modelo LLM (Gemini 2.5 Flash).
5. **Diseñar la arquitectura de despliegue** de la solución en producción.

---

## 📂 Estructura del Repositorio

```
Machine-learning-aplicado-al-sector-BPO/
│
├── README.md                                # Documentación del proyecto
├── requirements.txt                         # Dependencias del proyecto
├── .gitignore                               # Archivos excluidos del repositorio
├── .env.example                             # Ejemplo de variables de entorno (sin keys reales)
│
├── data/                                    # Datasets del proyecto
│   ├── New_HistConsultas.rar             # Historial de consultas (comprimido)
│   ├── New_Usuarios.rar                  # Características demográficas (comprimido)
│   └── comentarios_encuesta.rar            # Comentarios de encuestas de satisfacción
│
├── notebooks/                               # Notebooks del proyecto
│   ├── BPO_Clasificacion.ipynb              # EDA + Modelos predictivos supervisados
│   └── BPO_NLP_Satisfaccion.ipynb           # Análisis NLP de comentarios + Clustering
│
├── docs/                                    # Documentación técnica
│   └── arquitectura_despliegue.pdf          # Diseño de arquitectura de despliegue
│
└── images/                                  # Recursos visuales para el README
```

---

## 🔍 Metodología

### Notebook 1: EDA y Modelado Predictivo (`notebooks/BPO_Clasificacion.ipynb`)

#### 1.1 Análisis Exploratorio de Datos (EDA)

- Inspección de estructura, tipos de datos y valores faltantes en ambos datasets.
- Detección y eliminación de registros duplicados por `ID_Cuenta`.
- Merge de los datasets de consultas y usuarios.
- Análisis de distribución de variables numéricas (histogramas + KDE) y categóricas (countplots).
- Identificación de asimetría positiva en `Monto_adeudado`, `Duracion_llamada` y `Tiempo_en_espera`.
- Detección de outliers mediante boxplots.

**Hallazgo clave:** El cliente típico es soltero, entre 33-64 años, estrato 2-3, suscrito al Plan A, usuario de la app, no moroso, y con alta recomendación de marca. Los departamentos con mayor volumen de consultas son Valle del Cauca, Tolima y Santafé de Bogotá.

#### 1.2 Pruebas Estadísticas y Selección de Variables

Se aplicó un enfoque multinivel combinando pruebas paramétricas y no paramétricas:

| Tipo de Variable | Prueba | Propósito |
|---|---|---|
| Numérica | Pearson (Punto-Biserial) | Correlación lineal con target binario |
| Numérica | Spearman | Relaciones monótonas no lineales |
| Numérica | Mutual Information | Dependencias complejas no lineales |
| Categórica | Chi-Cuadrado (χ²) | Asociación global con el target |
| Categórica | Kruskal-Wallis | Diferencias en distribuciones entre categorías |
| Categórica | Mutual Information | Dependencia estadística general |

#### 1.3 Limpieza y Preprocesamiento

- Corrección de categorías erróneas (`/casado.` → `casado`, eliminación de `carlos`, `azzx456!`, `00000`).
- Imputación de nulos categóricos por moda.
- Normalización de nombres de departamentos.
- Tratamiento de outliers: eliminación por IQR + imputación por percentil 95.
- Codificación: **One-Hot Encoding** para variables con ≤5 categorías y **Target Encoding** para alta cardinalidad (`Motivo_llamada`, `Departamento`).

#### 1.4 Modelado Predictivo

Se evaluaron tres algoritmos de Gradient Boosting, todos con manejo de desbalance de clases (74.9% vs 25.1%):

| Modelo | Manejo de Desbalance | Optimización |
|---|---|---|
| **CatBoost** | `class_weights` (balanced) | Optuna (10 trials) |
| **LightGBM** | `scale_pos_weight` | Optuna (20 trials) |
| **XGBoost** | `scale_pos_weight` | Optuna (20 trials) |

Cada modelo fue evaluado con validación cruzada (5-Fold) y las métricas: **Accuracy, Sensibilidad (Recall), Especificidad, Precisión, F1-Score y ROC-AUC**.

---

### Notebook 2: Análisis NLP de Satisfacción (`notebooks/BPO_NLP_Satisfaccion.ipynb`)

#### 2.1 Preprocesamiento de Texto

- Limpieza y normalización de comentarios.
- Detección de textos pegados/duplicados internamente.
- Análisis de longitud y distribución de palabras.
- Nube de palabras inicial y top 20 palabras frecuentes.

#### 2.2 Clasificación con Gemini 2.5 Flash

- Prompt engineering estructurado para clasificar cada comentario en 10 categorías de insatisfacción.
- Análisis de sentimiento (muy negativo, negativo, neutro).
- Procesamiento por lotes con sistema de fallback de API Keys.
- Manejo robusto de errores y reintentos automáticos.

#### 2.3 Clustering Semántico

- Vectorización TF-IDF (unigramas + bigramas).
- Determinación de K óptimo con método del codo y Silhouette Score.
- K-Means clustering + visualización PCA.
- Validación cruzada: clusters vs categorías del LLM (heatmap).

#### 2.4 Diseño de Despliegue

Arquitectura completa en GCP documentada en [`docs/arquitectura_despliegue.pdf`](docs/arquitectura_despliegue.pdf).

---

## 📊 Resultados

### Modelo Predictivo Seleccionado

**LightGBM optimizado** fue seleccionado como el mejor modelo por:

- **Mejor balance sensibilidad-especificidad:** Mayor capacidad de detectar la clase minoritaria sin sacrificar precisión en la clase mayoritaria.
- **Mayor ROC-AUC** en validación cruzada tras optimización con Optuna.
- **Eficiencia computacional:** El crecimiento *leaf-wise* lo hace significativamente más rápido para reentrenamiento en producción.
- **Mayor estabilidad:** Menor dispersión entre folds, indicando robustez y menor riesgo de sobreajuste.

### Análisis NLP

- Los motivos de insatisfacción se concentran en problemas operativos: demoras, falta de respuesta y no resolución.
- El enfoque híbrido (LLM + clustering) permitió validar la clasificación y descubrir sub-patrones no contemplados.

---

## 🛠️ Stack Tecnológico

- **Lenguaje:** Python 3.x
- **Análisis de datos:** Pandas, NumPy
- **Visualización:** Matplotlib, Seaborn, WordCloud
- **Pruebas estadísticas:** SciPy (`chi2_contingency`, `kruskal`, `spearmanr`, `pointbiserialr`)
- **Feature Engineering:** Scikit-learn, Category Encoders (`TargetEncoder`)
- **Modelado:** CatBoost, LightGBM, XGBoost
- **Optimización:** Optuna
- **Evaluación:** Scikit-learn (`cross_validate`, `confusion_matrix`, `roc_auc_score`)
- **NLP / LLM:** Google Gemini 2.5 Flash API, TF-IDF
- **Clustering:** K-Means, PCA, Silhouette Score

---

## 🚀 Cómo Ejecutar

1. Clona el repositorio:
   ```bash
   git clone https://github.com/Mateo-Rua/Machine-learning-aplicado-al-sector-BPO.git
   cd Machine-learning-aplicado-al-sector-BPO
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Configura las API Keys para el notebook de NLP. Crea un archivo `.env` en la raíz del proyecto:
   ```bash
   cp .env.example .env
   ```
   Luego edita el `.env` con tus keys reales:
   ```
   GEMINI_API_KEY_1=tu_primera_api_key
   GEMINI_API_KEY_2=tu_segunda_api_key
   ```
   > ⚠️ **Importante:** El archivo `.env` está incluido en el `.gitignore` y nunca se sube al repositorio.

4. Ejecuta los notebooks:
   ```bash
   jupyter notebook notebooks/BPO_Clasificacion.ipynb
   jupyter notebook notebooks/BPO_NLP_Satisfaccion.ipynb
   ```

---

## 💡 Principales Conclusiones

1. **El perfil del cliente** está claramente definido: soltero, estrato 2-3, Plan A, usuario digital. Esto permite diseñar estrategias de atención segmentadas.

2. **El desbalance de clases** fue manejado exitosamente mediante ponderación de clases, evitando distorsiones por sobremuestreo.

3. **La selección de variables multinivel** (6 pruebas estadísticas) proporcionó una base rigurosa para alimentar los modelos con las features más informativas.

4. **LightGBM optimizado** es el modelo recomendado para producción por su rendimiento superior, eficiencia y estabilidad.

5. **El tratamiento híbrido de outliers** (eliminación + imputación por percentil 95) mejoró significativamente la generalización de los modelos.

6. **Los motivos de insatisfacción** se concentran en problemas operativos: demoras en solución, falta de respuesta y no resolución, lo que indica que el cuello de botella está en la capacidad de respuesta, no en la atención al cliente.

7. **El enfoque híbrido LLM + NLP clásico** (Gemini + TF-IDF/K-Means) demostró ser una estrategia robusta, donde el LLM aporta comprensión semántica y el clustering aporta validación estadística.

---

## 🔮 Trabajo Futuro

- Diseñar un pipeline de despliegue con MLflow para versionar y automatizar el reentrenamiento del modelo.
- Explorar *feature engineering* temporal con `Fecha_consulta` (estacionalidad, tendencias, días pico).
- Evaluar ensambles tipo *stacking* combinando los tres modelos.
- Reemplazar TF-IDF por embeddings densos (`text-embedding-004`) para clustering semántico más preciso.
- Fine-tunear un modelo LLM específico para el dominio BPO de energía.

---

## 👤 Autor

**Mateo Rua**

[![GitHub](https://img.shields.io/badge/GitHub-Mateo--Rua-181717?logo=github)](https://github.com/Mateo-Rua)

---

> *Proyecto de ciencia de datos aplicada al sector BPO, enfocado en la optimización de la experiencia del cliente mediante modelos predictivos de Machine Learning y análisis de lenguaje natural.*
