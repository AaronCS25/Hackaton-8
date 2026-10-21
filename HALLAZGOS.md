# Hallazgos para la exposición

Este documento resume las cifras y la narrativa del hackatón para construir la presentación. Todas las cifras provienen de `notebooks/03_analisis.ipynb` y las figuras están en `results/figuras`.

## Narrativa sugerida

1. Un agente orquestador necesita un router que lea mensajes informales en español y decida intención, agente, prioridad, aprobación humana y datos clave en un JSON. Un error de enrutamiento manda una emergencia a logística o ejecuta un reembolso sin supervisión.
2. Granite 4.2 3B sin ajuste no sirve para la tarea. En zero-shot inventa agentes que no existen y ningún JSON cumple el esquema (score 0.374). Con cinco ejemplos sube a 0.601.
3. Cualquier adapter entrenado lleva el JSON válido a 100% y el score a entre 0.935 y 0.958 en solo 3 minutos de entrenamiento en una A100.
4. El desempeño se satura en unos 117 mil parámetros entrenables, cerca de 0.003% del modelo. Más capacidad no mejora y puede desestabilizar el entrenamiento.
5. La superficie de respuesta predijo r = 9.8 y lr = 2.33e-4. La validación fuera de la grilla confirmó una meseta y no un pico.
6. Entre las tecnologías posteriores a LoRA, SHiRA fue la mejor y la más estable. Uni-LoRA logra casi lo mismo con 237 veces menos parámetros. FRoD y MiCA no justifican su uso en esta tarea.
7. El experimento completo costó 13.91 USD. Servir el modelo en lotes cuesta menos que una API comercial.

## Cifras clave

| Tema | Cifra |
|---|---|
| Dataset | 999 mensajes (707 train, 96 val, 196 test), 200 gold con casos difíciles |
| Acuerdo entre etiquetadores | kappa entre 0.977 y 1.000 en 150 mensajes |
| Zero-shot y few-shot | score 0.374 y 0.601 |
| Mejor modelo | SHiRA r = 32, lr = 1e-4, score 0.958 ± 0.002, exact match 0.755 |
| Mejor LoRA según validación | r = 32, lr = 2e-4, score 0.944 ± 0.016 |
| Modelo más eficiente | Uni-LoRA con 262,144 parámetros, score 0.938 |
| Codo de saturación | unos 117 mil parámetros entrenables |
| ANOVA LoRA | interacción r × lr con p < 0.001 y eta cuadrado de 0.539 |
| Divergencia | LoRA r = 128 con lr = 5e-4, score 0.320 |
| Óptimo predicho y validado | r = 9.8, lr = 2.33e-4, predicho 0.956 en val, observado 0.942 en val y 0.937 en test |
| Tiempo de entrenamiento | 3.2 a 3.4 min por corrida, 13.1 min en FRoD |
| VRAM de entrenamiento | 25 a 31 GB, 49 GB en FRoD |
| Latencia por solicitud | 1.9 a 2.1 s con el adapter fusionado, más de 86 s en FRoD sin fusionar |
| Cómputo total | 62 corridas, 11.6 horas-GPU en A100 80 GB, 13.91 USD facturados |
| Inferencia por cada mil solicitudes | 0.051 USD propio en lotes, 0.120 USD Claude Haiku 5.5, 2.410 USD Claude Sonnet 5.5 |

## Figuras

| Archivo | Uso en la presentación |
|---|---|
| `distribucion_clases.png` | Dataset y desbalance de agente y prioridad |
| `interaccion_rango_lr_lora.png` | Diseño factorial y divergencia en r = 128 |
| `superficie_respuesta_lora.png` | Óptimo predicho y punto de validación |
| `saturacion_parametros.png` | Frontera donde más capacidad deja de aportar |
| `calidad_vs_recursos.png` | Comparación de tecnologías por tiempo, VRAM y energía |
| `matriz_confusion_mejor_modelo.png` | Errores restantes del mejor modelo |

## Preguntas probables

### ¿Por qué Granite 4.2 3B?

Es un modelo denso de 3,660 millones de parámetros con licencia Apache 2.0 y soporte de español. Cabe holgadamente en una GPU de 24 GB para entrenar con adapters y se puede servir en una sola GPU.

### ¿Por qué estas tecnologías?

Cada una modifica un aspecto distinto de LoRA. MiCA cambia el subespacio donde se aprende, FRoD busca actualizaciones de rango completo, SHiRA entrena una fracción dispersa de los pesos originales y Uni-LoRA comparte un único vector entrenable entre todas las capas. Las cuatro están implementadas en PEFT 0.21.2.

### ¿Qué tan confiables son las etiquetas?

Las etiquetas de agente, prioridad y aprobación se derivan con reglas deterministas, por lo que son consistentes por construcción. Un segundo etiquetado independiente de 150 mensajes obtuvo un kappa entre 0.977 y 1.000, lo que muestra que las reglas de la guía son precisas y reproducibles.

### ¿Por qué el óptimo validado no superó a la grilla?

Dentro de la meseta las diferencias entre configuraciones son del mismo orden que la variación entre semillas (desviación media de 0.009). La validación confirma que la región óptima es amplia y que seguir ajustando r o lr no produce mejoras medibles.

### ¿Por qué FRoD tarda tanto?

Antes de entrenar descompone conjuntamente las matrices de pesos de las 40 capas, lo que tomó entre 5 y 26 minutos en CPU. Además aplica rotaciones dispersas en cada paso, lo que cuadruplica el tiempo de entrenamiento. Sin fusionar el adapter, cada solicitud tarda más de 86 segundos.

### ¿Por qué evaluar con el adapter fusionado?

Es la forma de despliegue en producción y elimina la sobrecarga del adapter. Con todos los adapters fusionados la latencia es la del modelo base.

### ¿Conviene entrenar un modelo propio o usar una API?

En costo, el modelo propio servido en lotes es más barato que Claude Haiku 5.5 y mucho más barato que Claude Sonnet 5.5. La ventaja desaparece si la GPU queda ociosa. La calidad de las APIs en esta tarea no se midió.
