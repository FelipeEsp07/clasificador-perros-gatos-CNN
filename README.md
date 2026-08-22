# Clasificador de perros y gatos — CNN

Clasificador binario de imágenes (perro vs. gato) construido con la
arquitectura **ResNet18**, entrenada **completamente desde cero** (sin
pesos preentrenados de ImageNet, sin transfer learning), usando PyTorch.
Incluye entrenamiento con early stopping, evaluación con métricas completas
y una interfaz web en Streamlit para probar el modelo con imágenes propias.

## Por qué entrenar desde cero

Se eligió entrenar ResNet18 completamente desde cero (`weights=None`, sin
pesos de ImageNet) para estudiar el comportamiento de una CNN sin transfer
learning: cuánto tarda en converger, qué learning rate y cuántas épocas hacen
falta cuando el modelo debe aprender desde los datos disponibles tanto las
características de bajo nivel (bordes, texturas) como las de alto nivel
(formas de perros/gatos), sin ningún conocimiento previo. Es esperable que,
con un dataset de solo ~8,700 imágenes de entrenamiento, el modelo converja
más lento y le cueste más generalizar que una red que parte de pesos
preentrenados.

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
de Kaggle (~10,000 imágenes), que ya viene dividido en:

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

**Nota sobre normalización**: como el modelo no usa pesos preentrenados,
las imágenes NO se normalizan con las estadísticas de ImageNet (`mean`/`std`).
En su lugar se usa un escalado simple a `[0, 1]`, que es lo que hace
`transforms.ToTensor()` por sí solo.

## Entrenar el modelo

```powershell
venv\Scripts\activate
python src\train.py
```

El entrenamiento corre en un único loop (no hay fases de backbone
congelado/descongelado, porque no hay pesos preentrenados que proteger):
todas las capas de ResNet18 se ajustan desde la época 1, con un learning
rate inicial de `1e-3` (típico de entrenar una red convolucional desde
cero), hasta 30 épocas máximo y early stopping con paciencia de 6 épocas
monitoreando `val_loss`. Siempre se guarda el mejor checkpoint en
`models/best_model.pth`.

**Weights & Biases es opcional.** Si `wandb` está instalado (viene en
`requirements.txt`) y hay sesión iniciada, cada corrida se registra
automáticamente en el proyecto `clasificador-perros-gatos`; los campos
`backbone: "resnet18"` y `pretrained: false` en el config de cada run
permiten filtrar/agrupar por arquitectura y por si usó o no transfer
learning. Si no quieres usarlo, el entrenamiento funciona igual sin él:
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

## Diseño del entrenamiento

| Aspecto | Valor |
|---|---|
| Pesos iniciales | `weights=None` (aleatorios) |
| Normalización de entrada | Escalado simple a `[0, 1]` |
| Estrategia de entrenamiento | 1 fase: todo entrenable desde el inicio |
| Learning rate | 1e-3 (constante, con scheduler adaptativo) |
| Épocas máximas | 30 |
| Paciencia early stopping | 6 |
| Parámetros entrenables al inicio | 11,177,025 de 11,177,025 (100%) |

Al no haber pesos preentrenados que proteger, no hay fases de
congelado/descongelado: todas las capas se entrenan juntas desde la época 1
con un único optimizador (`AdamW`), scheduler (`ReduceLROnPlateau`),
checkpointing por mejor `val_loss` y criterio de pérdida
(`BCEWithLogitsLoss`).

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
  más épocas para alcanzar un desempeño razonable, dado que debe aprender
  desde cero tanto las características de bajo nivel como las de alto nivel
  a partir de un dataset relativamente pequeño (~8,700 imágenes).
