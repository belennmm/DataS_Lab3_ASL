
> [!TIP]
> Debido al peso del data set original, este no fue subido al repositorio. Si desea compilarlo debe de descargar la data original en el siguiente link https://www.kaggle.com/datasets/grassknoted/asl-alphabet


# Laboratorio 3 — Deep Learning: Reconocimiento de Lenguaje de Señas (ASL)

**Curso:** CC3084 – Data Science
**Universidad del Valle de Guatemala** — Facultad de Ingeniería, Departamento de Ciencias de la Computación
**Semestre II – 2026**

**Integrantes:** Belén Monterroso 231497 , Melisa Mendizabal 23778 , Renato Rojas 23813 

## Contexto

SignBridge es una startup guatemalteca de tecnología inclusiva que desarrolla un traductor de Lenguaje de Señas Americano (ASL) en tiempo real para apps educativas. Este laboratorio construye y evalúa el primer prototipo del motor de reconocimiento: un clasificador de letras del alfabeto ASL a partir de imágenes de manos.

Dataset utilizado: [ASL Alphabet (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) — ~87,000 imágenes de entrenamiento en 29 clases (A-Z, más `space`, `del`, `nothing`).

## Estructura del repositorio

```
├── Lab3_DataScience.pdf          # PDF de informe
├── procesamiento.py              # Funciones auxiliares (hashing, deduplicación, split train/val/test)
├── lab3_EDA.ipynb                # Análisis exploratorio de datos
├── lab3_cnn.ipynb                # Preprocesamiento de imágenes + modelos CNN y red neuronal simple
├── lab3_SVM_transformacion.ipynb # Modelo con otro algoritmo (SVM) + image augmentation
├── evaluacion_propia.ipynb       # Prueba del mejor modelo con fotos propias del equipo
├── mejor_modelo_asl.keras        # Modelo entrenado con mejor desempeño (guardado)
├── codificador_asl.pkl           # LabelEncoder ajustado (índices <-> letras)
├── TestPropio/                   # Fotos propias del equipo para pruebas (una carpeta por integrante)
```

## Contenido por notebook

### `lab3_EDA.ipynb`
Análisis exploratorio del dataset: ejemplos de letras, distribución de clases, resolución y formato de las imágenes, identificación de letras visualmente similares (confusión potencial, p. ej. M/N/S o U/V/R).

### `lab3_cnn.ipynb`
- **Preprocesamiento:** carga de imágenes, submuestreo estratificado por clase, eliminación de duplicados (perceptual hashing + verificación por MSE), split train/val/test (70/15/15), resize a 64x64 y normalización a `[0, 1]`.
- **Modelos de Deep Learning:**
  - Red neuronal simple (fully-connected) — modelo base.
  - CNN #1 — arquitectura base (2 bloques convolucionales).
  - CNN #2 — arquitectura más profunda, con variaciones de hiperparámetros (dropout, learning rate, batch size).
- **Evaluación:** accuracy, loss, matriz de confusión, tiempo de entrenamiento, comparación entre los 3 modelos.
- Guarda el mejor modelo (`mejor_modelo_asl.keras`) y el encoder (`codificador_asl.pkl`) para su uso en otros notebooks.

### `lab3_SVM_transformacion.ipynb`
Modelo con un algoritmo alternativo (SVM) y aplicación de transformaciones/image augmentation a los datos. Se compara contra los modelos de CNN; como era esperable dado que SVM no captura relaciones espaciales tan bien como una CNN, su desempeño resultó inferior al de la mejor CNN (CNN #2).

### `evaluacion_propia.ipynb`
Carga el modelo y encoder guardados, y evalúa el desempeño del mejor modelo con fotos propias tomadas por el equipo.

## Ranking de resultados

| Modelo | Test Accuracy |
|---|---|
| CNN #2 | **Mejor modelo** — arquitectura más profunda 
| CNN #1 | Arquitectura convolucional base 
| SVM  | Peor desempeño que las CNN 
| Red neuronal simple (FC) | Modelo base; **el peor de todos** 

