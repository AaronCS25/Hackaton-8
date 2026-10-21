# Router de solicitudes con Granite 4.2 3B y PEFT

Este repositorio ajusta `ibm-granite/granite-4.2-3b` para que funcione como router de un agente orquestador de una agencia peruana de viajes y eventos. El modelo recibe el mensaje informal de un cliente y devuelve un JSON con la intención (8 clases), el agente especializado que debe atenderlo (5 clases), la prioridad (4 clases), la necesidad de aprobación humana y cuatro slots (destino, fecha, número de personas y monto).

El trabajo compara LoRA con cuatro tecnologías posteriores implementadas en Hugging Face PEFT, que son FRoD, MiCA, Uni-LoRA y SHiRA. El diseño experimental incluye un factorial completo de LoRA, tres niveles de capacidad por tecnología, dos réplicas con semillas distintas y la validación del punto óptimo predicho por una superficie de respuesta.

## Resultados principales

Score en test (promedio del macro-F1 de intención, agente y prioridad) de la mejor configuración de cada tecnología según validación, con la media de dos réplicas.

| Modelo | Parámetros entrenables | Score test | JSON válido | Exact match | Entrenamiento |
|---|---|---|---|---|---|
| Zero-shot | 0 | 0.374 | 0.0% | 0.000 | |
| Few-shot (5 ejemplos) | 0 | 0.601 | 92.9% | 0.122 | |
| SHiRA r = 32 | 62.3 M | 0.958 | 100% | 0.755 | 3.3 min |
| LoRA r = 32 | 62.3 M | 0.944 | 100% | 0.763 | 3.3 min |
| Uni-LoRA θ = 262,144 | 0.26 M | 0.938 | 100% | 0.699 | 3.3 min |
| FRoD sparse_rate = 0.01 | 43.5 M | 0.936 | 100% | 0.686 | 13.1 min |
| MiCA r = 32 | 30.1 M | 0.935 | 100% | 0.656 | 3.2 min |

1. El desempeño se satura en unos 117 mil parámetros entrenables. Por encima de ese punto, aumentar la capacidad del adapter no mejora el score.
2. En LoRA, el rango y el learning rate interactúan de forma significativa (ANOVA, p < 0.001). Con r = 128 y lr = 5e-4 el entrenamiento diverge.
3. La superficie de respuesta predijo un óptimo en r = 9.8 y lr = 2.33e-4. La validación en r = 10 y lr = 2.3e-4 obtuvo 0.937 en test, dentro de la meseta de las mejores configuraciones.
4. SHiRA obtuvo el mejor score y la menor variación entre semillas. Uni-LoRA alcanza casi la misma calidad con 237 veces menos parámetros. FRoD tiene una calidad comparable con cuatro veces más tiempo de entrenamiento y 49 GB de VRAM.
5. Las 62 corridas consumieron 11.6 horas-GPU en A100 de 80 GB y costaron 13.91 USD en vast.ai.

El análisis completo está en `notebooks/03_analisis.ipynb` y las figuras en `results/figuras`.

## Estructura

| Ruta | Contenido |
|---|---|
| `docs/guia_etiquetado.md` | Definiciones de intención, atributos, reglas derivadas y slots |
| `data/raw` | 1000 mensajes en diez lotes, con intención, atributos y slots |
| `data/processed` | Splits train (707), val (96) y test (196) con el JSON objetivo |
| `data/reetiquetado` | Re-etiquetado independiente de 150 mensajes para medir el acuerdo |
| `data/precios_referencia.json` | Precios de GPU y de API usados en el análisis de costos |
| `src/router` | Pipeline de entrenamiento, evaluación, métricas, grilla experimental y análisis |
| `scripts` | Validación del dataset, construcción de splits y ejecución de la cola de corridas |
| `notebooks` | 01 dataset, 02 entrenamiento y evaluación, 03 análisis de resultados |
| `results/runs` | Un JSON por corrida con configuración, métricas, recursos y tiempos |
| `results/preds` | Predicciones crudas de cada corrida en validación y test |
| `results/figuras` | Figuras principales exportadas de los notebooks |

## Reproducción

El entorno se gestiona con uv. Las versiones están fijadas en `uv.lock` (torch 2.11.0 con CUDA 12.8, transformers 5.20.2 y peft 0.21.2).

```bash
uv sync
uv run python scripts/validar_dataset.py data/raw/*.jsonl
uv run python scripts/construir_splits.py
```

Las corridas medidas se ejecutaron en dos GPUs A100 SXM4 de 80 GB, una corrida por GPU.

```bash
export ROUTER_GRADIENT_CHECKPOINTING=0 ROUTER_MICRO_BATCH=4 ROUTER_GRAD_ACCUM=4
bash scripts/correr_bloques.sh 2 base piloto barrido_lr
bash scripts/lanzar_principal.sh 2
uv run python scripts/cola.py --bloque lora --gpus 2
uv run python scripts/cola.py --bloque validacion --r 10 --lr 0.00023 --gpus 2
```

La cola omite las corridas que ya tienen resultado, por lo que puede reanudarse. En GPUs de menos de 24 GB se recomienda mantener el gradient checkpointing activado (valor por defecto) con micro-batch 8 y acumulación 2.

## Construcción de los datos

Los mensajes y sus anotaciones se construyeron siguiendo la guía de etiquetado. Ocho lotes cubren subdominios distintos y dos lotes gold se concentran en casos difíciles. Las etiquetas de agente, prioridad y aprobación se derivan con reglas deterministas. Un segundo etiquetado independiente de 150 mensajes, sin acceso a las etiquetas originales, obtuvo un kappa de Cohen entre 0.977 y 1.000 según el campo.
