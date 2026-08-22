# Clasificador de perros y gatos — CNN

Clasificador binario de imágenes (perro vs. gato) construido con la
arquitectura **ResNet18**, entrenada **completamente desde cero** (sin
pesos preentrenados de ImageNet, sin transfer learning), usando PyTorch.
Incluye entrenamiento con early stopping, evaluación con métricas completas
y una interfaz web en Streamlit para probar el modelo con imágenes propias.

## ¿Qué es una CNN (Red Neuronal Convolucional)?

Una **red neuronal convolucional (CNN)** es una arquitectura de red neuronal
diseñada específicamente para procesar datos con estructura de rejilla, como
imágenes (una matriz de píxeles). A diferencia de una red totalmente
conectada (donde cada neurona de una capa se conecta con absolutamente todas
las neuronas de la capa anterior), una CNN está construida sobre dos ideas
que la hacen mucho más eficiente y efectiva para visión por computador:

- **Localidad**: en una imagen, la información relevante suele estar
  concentrada en regiones pequeñas y vecinas (un borde, una textura, un ojo).
  En vez de conectar cada neurona con toda la imagen, una **capa
  convolucional** aplica pequeños filtros (llamados **kernels**, típicamente
  de 3×3 o 5×5 píxeles) que solo "miran" una región local de la imagen a la
  vez, y van deslizándose (convolucionando) sobre toda la imagen para
  producir un **mapa de características** (*feature map*). Cada kernel
  aprende a detectar un patrón visual concreto: bordes horizontales,
  cambios de color, texturas, etc.
- **Weight sharing (pesos compartidos)**: el mismo kernel se reutiliza en
  todas las posiciones de la imagen, en vez de aprender pesos distintos para
  cada píxel. Esto reduce drásticamente la cantidad de parámetros a
  entrenar (comparado con una red totalmente conectada del mismo tamaño de
  entrada) y le da a la red una propiedad clave: si aprende a detectar una
  oreja de gato en la esquina superior izquierda de una imagen, reconoce esa
  misma oreja aunque aparezca en cualquier otra posición (**invarianza a la
  traslación**).

Además de las capas convolucionales, las CNN suelen incluir capas de
**pooling** (por ejemplo *max pooling*), que reducen el tamaño espacial de
los mapas de características quedándose con el valor más representativo
(el máximo) de cada región pequeña. Esto cumple dos propósitos: reduce el
costo computacional de las capas siguientes y aporta un poco más de
robustez ante pequeñas traslaciones o deformaciones de la imagen.

Al apilar muchas capas convolucionales, la red construye una **jerarquía de
representaciones**: las primeras capas detectan patrones simples (bordes,
esquinas, manchas de color), las capas intermedias combinan esos patrones en
formas más complejas (texturas de pelaje, contornos de orejas u ojos), y las
capas finales combinan todo eso en conceptos de alto nivel (la forma general
de un perro o un gato). Por esta combinación de eficiencia (menos
parámetros gracias al weight sharing), sensibilidad a la estructura espacial
de la imagen (gracias a la localidad) y capacidad de aprender jerarquías de
patrones, las CNN son desde hace más de una década la arquitectura estándar
para tareas de visión por computador como la clasificación de imágenes que
resuelve este proyecto.

## Arquitectura ResNet en detalle

**ResNet** (*Residual Network*, presentada por He et al. en 2015) es una
familia de arquitecturas de CNN que resolvió un problema muy concreto:
antes de ResNet, apilar más y más capas convolucionales para hacer redes
más "profundas" dejaba de ayudar a partir de cierto punto — y no por
sobreajuste, sino por un problema de optimización conocido como
**degradación del gradiente** (*degradation problem*). En redes muy
profundas, el error de entrenamiento (no solo el de validación) empezaba a
empeorar al agregar más capas, porque el gradiente que se propaga hacia
atrás durante el entrenamiento (backpropagation) se iba atenuando o
distorsionando capa a capa, y a la red le costaba cada vez más aprender
incluso la función identidad (es decir, "no transformar nada") cuando eso
era lo óptimo para una capa en particular.

La solución de ResNet son las **conexiones residuales** (*skip
connections* o *shortcut connections*): en vez de que cada bloque de capas
aprenda directamente la transformación deseada `H(x)`, se le pide que
aprenda solo el **residuo** `F(x) = H(x) - x`, y la salida real del bloque
se calcula sumando la entrada original sin modificar:

```
salida = F(x) + x
```

Esta suma se implementa literalmente como una conexión que "salta" el
bloque de capas convolucionales y se suma a su salida. La ventaja es doble:

1. **Le facilita a la red aprender la identidad cuando conviene.** Si la
   transformación óptima para un bloque es no hacer nada, basta con que
   `F(x)` converja a cero — mucho más fácil de aprender para un conjunto de
   capas que aprender la identidad exacta desde una inicialización aleatoria.
2. **Mejora el flujo del gradiente durante el entrenamiento.** La conexión
   residual le da al gradiente un "camino corto" para propagarse hacia atrás
   sin atenuarse a través de todas las capas intermedias, lo que en la
   práctica permite entrenar redes de decenas o incluso cientos de capas sin
   que el rendimiento se degrade al hacerlas más profundas.

**ResNet18** es la variante más pequeña de esta familia con 18 capas con
peso (de ahí el nombre): una capa convolucional inicial de 7×7, seguida de
4 grupos de **bloques residuales básicos** (*BasicBlock*, cada uno con dos
capas convolucionales de 3×3 y su propia conexión residual sumando la
entrada del bloque a su salida), y finalmente una capa totalmente conectada
de clasificación. Es una arquitectura relativamente ligera dentro de la
familia ResNet (frente a variantes más profundas como ResNet50 o
ResNet101), lo que la hace razonable para entrenar desde cero con un
dataset de tamaño moderado como el de este proyecto.

En este proyecto se usa exactamente esa arquitectura, instanciada con
`torchvision.models.resnet18(weights=None)` (ver `src/model.py`): el `None`
en `weights` es lo que indica que **no** se cargan los pesos preentrenados
en ImageNet, sino que todos los parámetros parten de una inicialización
aleatoria, tal como se explica en la siguiente sección. La única
modificación sobre la ResNet18 estándar es reemplazar su cabeza de
clasificación original (pensada para 1000 clases de ImageNet) por una
cabeza binaria (`Dropout` + una capa lineal a un solo logit), adecuada para
distinguir entre solo dos clases: perro y gato.

Para dejarlo explícito: **este es un modelo convolucional (CNN) que usa la
arquitectura ResNet** — todo lo descrito en la sección anterior sobre
capas convolucionales, kernels y pooling aplica aquí, con la adición de las
conexiones residuales que caracterizan específicamente a ResNet.

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
