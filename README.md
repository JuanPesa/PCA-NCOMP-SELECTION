# Análisis PCA y Clasificación con Redes Densas en Datasets de Imágenes

Este repositorio contiene un script de Python desarrollado en Google Colab que explora la aplicación de Principal Component Analysis (PCA) para la reducción de dimensionalidad en tres datasets populares de imágenes: MNIST, Digits y Fashion MNIST. Posteriormente, se entrena una red neuronal densa (Dense Neural Network) para la clasificación en cada dataset, evaluando el impacto de diferentes criterios de selección de componentes PCA en la precisión y el tiempo de entrenamiento.

## Contenido

-   `PCA NCOMP SELECTION.ipynb`: El script principal de Python que contiene todo el código para la carga de datos, preprocesamiento, aplicación de PCA (con diferentes criterios), entrenamiento de la red densa y visualización de resultados.
-   `README.md`: Este archivo.

## Datasets Utilizados

El script trabaja con los siguientes datasets, los cuales son cargados directamente utilizando `sklearn.datasets.fetch_openml` y `sklearn.datasets.load_digits`:

1.  **MNIST**: Dataset de dígitos escritos a mano (70,000 muestras, 784 características). 
2.  **Digits**: Dataset más pequeño de dígitos (1,797 muestras, 64 características). 
3.  **Fashion MNIST**: Dataset de imágenes de artículos de moda (70,000 muestras, 784 características). 

## Preprocesamiento y PCA

Antes de aplicar PCA y entrenar los modelos, los datos pasan por los siguientes pasos:

1.  **Carga de datos**: Los datasets se cargan y las etiquetas se convierten a `int32` para asegurar la compatibilidad con los modelos de Keras. 
2.  **Escalado**: Los datos de las características (`X`) son escalados utilizando `StandardScaler` para que tengan media 0 y varianza 1, un paso crucial para PCA. 
3.  **Aplicación de PCA**: Se aplica PCA para reducir la dimensionalidad de los datos. Se evalúan diferentes criterios para seleccionar el número óptimo de componentes principales:
    * **Criterio de Kaiser**: Retiene componentes con valores propios (eigenvalues) mayores que la media de todos los valores propios. 
    * **Varianza Explicada**: Retiene el número de componentes que explican un porcentaje de varianza (por defecto, 95%). 
    * **Análisis Paralelo de Horn**: Compara los valores propios reales con los de datos aleatorios para determinar los componentes significativos. 
    * Además, se incluyen pruebas con el **número total de características** y la **mitad de las características originales** para fines comparativos. 

## Red Neuronal Densa

Una red neuronal densa simple de Keras se utiliza para la clasificación de imágenes después de la reducción de dimensionalidad con PCA.

-   **Arquitectura del Modelo**:
    * Capa de entrada con `n_components` características. 
    * Una capa `Dense` de 256 neuronas con activación `relu`. 
    * Una capa `Dropout` con tasa de 0.3. 
    * Una capa `Dense` de 128 neuronas con activación `relu`. 
    * Una segunda capa `Dropout` con tasa de 0.3. 
    * Una capa `Dense` de salida con `num_classes` neuronas y activación `softmax`. 
-   **Compilación**: Optimizador `adam`, función de pérdida `sparse_categorical_crossentropy` y métrica `accuracy`. 
-   **Entrenamiento**: Se entrena el modelo por hasta 100 épocas con un tamaño de lote de 64, utilizando un `validation_split` del 15% y `EarlyStopping` (patience=5) para prevenir el sobreajuste. 

## Resultados y Análisis

El script imprime los resultados de la aplicación de cada criterio de PCA (número de componentes) y la precisión final en el conjunto de prueba para cada combinación de dataset y criterio. También genera gráficas de precisión de entrenamiento y validación para cada modelo individual y gráficas comparativas de la precisión de validación para todos los criterios por dataset.

### Ejemplos de Resultados (MNIST):

-   **Características originales**: 784 
-   **Criterio de Kaiser**: 179 componentes 
-   **Varianza Acumulada (umbral 0.95)**: 332 componentes 
-   **Análisis Paralelo de Horn**: 150 componentes 

| Criterio                     | Componentes | Precisión Final (MNIST) | Tiempo de Entrenamiento (MNIST) |
| :--------------------------- | :---------- | :---------------------- | :------------------------------ |
| Todas las Características    | 784         | 0.9668        | 69.06 segundos        |
| Mitad de Características     | 392         | 0.9699        | 42.45 segundos        |
| Kaiser                       | 179         | 0.9742        | 59.22 segundos        |
| Varianza Explicada (95%)     | 332         | 0.9734        | 70.56 segundos        |
| Análisis Paralelo            | 150         | 0.9761        | 70.16 segundos        |

Los resultados muestran cómo la reducción de dimensionalidad con PCA, utilizando criterios específicos, puede mantener o incluso mejorar la precisión de clasificación mientras reduce significativamente el número de características de entrada y, en algunos casos, el tiempo de entrenamiento. 

## Cómo Ejecutar el Script

Este script está diseñado para ejecutarse en un entorno de Google Colab.

1.  Abre Google Colab.
2.  Ve a `Archivo` > `Subir cuaderno` y selecciona el archivo `PCA NCOMP SELECTION.ipynb`.
3.  Asegúrate de tener un entorno de ejecución con GPU si el tiempo de entrenamiento es una preocupación (aunque para estos datasets y modelos, CPU suele ser suficiente).
4.  Ejecuta todas las celdas secuencialmente.

## Librerías Necesarias

Asegúrate de que tu entorno de Python tenga instaladas las siguientes librerías:

-   `numpy`
-   `pandas`
-   `matplotlib`
-   `scikit-learn`
-   `tensorflow`
-   `keras`

(Todas estas librerías suelen venir preinstaladas en Google Colab). 

---
