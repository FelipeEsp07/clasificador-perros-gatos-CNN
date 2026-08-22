# Clasificador de perros y gatos — CNN

Clasificador binario de imágenes (perro vs. gato) construido con la
arquitectura **ResNet18**, entrenada **completamente desde cero** (sin
pesos preentrenados de ImageNet, sin transfer learning), usando PyTorch.
Incluye entrenamiento con early stopping, evaluación con métricas completas
y una interfaz web en Streamlit para probar el modelo con imágenes propias.

## Propósito comparativo de este proyecto

Este proyecto es parte de un experimento de tres puntas para entender qué
aporta el transfer learning en esta tarea:

1. [`clasificador-perros-gatos`](../clasificador-perros-gatos) —
   **EfficientNet-B0 + transfer learning** (backbone preentrenado en
   ImageNet, fine-tuning progresivo en dos fases). Resultado de referencia:
   97.33% accuracy en test.
2. **Este proyecto** — **ResNet18 entrenado desde cero** (misma arquitectura
   ResNet18 que se usó inicialmente con transfer learning en una versión
   anterior de este mismo directorio, pero ahora con pesos inicializados
   al azar, `weights=None`, y sin fases de congelado/descongelado).

La comparación relevante aquí NO es "¿qué arquitectura es mejor?" (eso
requeriría entrenar ambas arquitecturas en igualdad de condiciones, con o
sin transfer learning para ambas). La comparación es **"¿cuánto aporta el
transfer learning frente a aprender todo desde los datos disponibles?"**,
usando ResNet18 como arquitectura de prueba. Se espera que el modelo
entrenado desde cero rinda notablemente peor que la variante con transfer
learning, precisamente porque el dataset de ~8,700 imágenes de
entrenamiento es pequeño para que una red convolucional aprenda desde cero
representaciones visuales tan buenas como las que ya trae ImageNet
(~1.2 millones de imágenes).

## Estructura del proyecto

```
clasificador-perros-gatos-CNN/
├── requirements.txt
├── data/                       # dataset (no versionado en git)
├── src/
│   ├── dataset.py              # Dataset/DataLoader, transforms, validación de imágenes
│   ├── model.py                 # ResNet18 (weights=None) + cabeza binaria
│   ├── train.py                 # entrenamiento en un solo loop, checkpointing, W&B
│   └── evaluate.py              # métricas finales sobre el test set
├── models/
│   ├── best_model.pth           # checkpoint del mejor modelo (generado por train.py)
│   └── matriz_confusion.png     # generado por evaluate.py
└── app/
    └── streamlit_app.py         # interfaz de usuario
```

## Requisitos

- Python 3.12
- Una GPU NVIDIA con CUDA es opcional pero muy recomendable (el entrenamiento
  en CPU es considerablemente más lento). Este proyecto se probó con
  `torch`/`torchvision` compilados para CUDA 12.1.

## Configuración del entorno

```powershell
cd clasificador-perros-gatos-CNN
python -m venv venv
venv\Scripts\activate

# Con GPU NVIDIA (CUDA 12.1):
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Sin GPU (solo CPU): editar requirements.txt y quitar "+cu121" de torch/torchvision
# antes de correr pip install -r requirements.txt, o instalar manualmente:
# pip install torch torchvision
```

## Dataset

El proyecto usa el dataset [`tongpython/cat-and-dog`](https://www.kaggle.com/datasets/tongpython/cat-and-dog)
de Kaggle (~10,000 imágenes), el mismo que usa `clasificador-perros-gatos`,
que ya viene dividido en:

```
data/training_set/training_set/{cats,dogs}/
data/test_set/test_set/{cats,dogs}/
```

Para descargarlo con la CLI oficial de Kaggle (requiere `~/.kaggle/kaggle.json` configurado):

```powershell
kaggle datasets download -d tongpython/cat-and-dog -p data --unzip
```

`src/dataset.py` valida automáticamente las imágenes al construir el dataset:
mueve a una subcarpeta `_invalidas` cualquier archivo que no sea una imagen
legible (por ejemplo, el archivo `_DS_Store` que trae el ZIP de Kaggle por
ser residuo de macOS).

**Nota sobre normalización**: a diferencia de las variantes con transfer
learning, aquí las imágenes NO se normalizan con las estadísticas de
ImageNet (`mean`/`std`), porque el modelo no usa pesos preentrenados con ese
preprocesamiento. En su lugar se usa un escalado simple a `[0, 1]`, que es
lo que hace `transforms.ToTensor()` por sí solo.

## Entrenar el modelo

```powershell
venv\Scripts\activate
python src\train.py
```

El entrenamiento corre en un único loop (no hay fases de backbone
congelado/fine-tuning, porque no hay pesos preentrenados que proteger):
todas las capas de ResNet18 se ajustan desde la época 1, con un learning
rate inicial de `1e-3` (más alto que el usado en fine-tuning), hasta 30
épocas máximo y early stopping con paciencia de 6 épocas monitoreando
`val_loss`. Siempre se guarda el mejor checkpoint en `models/best_model.pth`.

**Weights & Biases es opcional.** Si `wandb` está instalado (viene en
`requirements.txt`) y hay sesión iniciada, cada corrida se registra
automáticamente en el mismo proyecto `clasificador-perros-gatos` que usan
las variantes con transfer learning (a propósito, para poder comparar los
tres runs directamente desde el dashboard); los campos `backbone: "resnet18"`
y `pretrained: false` en el config de cada run permiten filtrar/agrupar por
arquitectura y por si usó o no transfer learning. Si no quieres usarlo, el
entrenamiento funciona igual sin él:
- Quita `wandb` de `requirements.txt` antes de instalar (o no lo instales) —
  `train.py` lo detecta y sigue sin registrar métricas, sin romperse.
- O déjalo instalado pero corre con `WANDB_MODE=disabled`
  (`$env:WANDB_MODE="disabled"` en PowerShell) para evitar el login.

## Evaluar el modelo

```powershell
venv\Scripts\activate
python src\evaluate.py
```

Imprime accuracy, precision, recall y F1 sobre `test_set` (el subconjunto
que nunca se usó durante el entrenamiento) y guarda la matriz de confusión
en `models/matriz_confusion.png`.

## Correr la app de Streamlit

El repositorio ya incluye el checkpoint entrenado en `models/best_model.pth`,
así que no hace falta correr `train.py` primero — clona, instala las
dependencias y corre la app directamente.

```powershell
venv\Scripts\activate
streamlit run app\streamlit_app.py
```

Se abre en el navegador (por defecto `http://localhost:8501`). Sube una
imagen JPG o PNG y la app muestra la clase predicha (perro/gato) junto con
el porcentaje de confianza.

## Diferencias frente a `clasificador-perros-gatos` (EfficientNet-B0 + transfer learning)

| Aspecto | EfficientNet-B0 (transfer learning) | ResNet18 (desde cero) |
|---|---|---|
| Pesos iniciales | `EfficientNet_B0_Weights.DEFAULT` (ImageNet) | `weights=None` (aleatorios) |
| Normalización de entrada | Estadísticas de ImageNet | Escalado simple a `[0, 1]` |
| Estrategia de entrenamiento | 2 fases: cabeza congelada → fine-tuning | 1 fase: todo entrenable desde el inicio |
| Learning rate | 1e-3 (cabeza) → 1e-5 (fine-tuning) | 1e-3 (constante, con scheduler adaptativo) |
| Épocas máximas | 12 + 10 = 22 | 50 |
| Paciencia early stopping | 3 | 6 |
| Parámetros entrenables al inicio | 1,281 de ~4.0M | 11,177,025 de 11,177,025 (100%) |

Todo lo demás — dataset, augmentation, split, criterio de pérdida
(`BCEWithLogitsLoss`), optimizador (`AdamW`), scheduler
(`ReduceLROnPlateau`), checkpointing por mejor `val_loss`, semillas, métricas
de evaluación y la interfaz de Streamlit — sigue el mismo patrón que el
resto de proyectos de esta comparación.

## Notas de implementación

- El modelo produce un único logit (no una probabilidad) y se entrena con
  `nn.BCEWithLogitsLoss`, más estable numéricamente que aplicar `Sigmoid` +
  `BCELoss` por separado.
- `num_workers=0` en los `DataLoader`: en Windows, `DataLoader` crea
  procesos hijos en modo `spawn`, que reimportan PyTorch/SciPy/scikit-learn
  desde cero en cada worker. En la máquina de desarrollo eso agotó el
  archivo de paginación del sistema y los workers fallaban con
  `ImportError: DLL load failed`. Con `num_workers=0` la carga de datos
  ocurre en el proceso principal.
- Entrenar desde cero converge sustancialmente más lento que hacer fine-tuning
  sobre un backbone preentrenado: se espera que este modelo necesite muchas
  más épocas para alcanzar un desempeño razonable, y es normal (y esperado)
  que su accuracy final en test quede por debajo del 97.33% de la variante
  con transfer learning — esa diferencia es precisamente lo que este
  experimento busca cuantificar, no un indicio de que ResNet18 sea una
  arquitectura inferior a EfficientNet-B0 en igualdad de condiciones.
