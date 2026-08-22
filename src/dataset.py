"""
Carga y preprocesamiento de datos para el clasificador binario perro/gato.

Este módulo se encarga de:
- validar imágenes corruptas o archivos que no son imágenes antes de construir
  el dataset (el ZIP de Kaggle trae residuos de macOS como `_DS_Store` dentro
  de las carpetas de clases, que romperían la carga si no se filtran antes).
- definir un `Dataset` de PyTorch que lee imágenes de disco bajo demanda.
- definir los transforms de entrenamiento (con aumento de datos) y de
  evaluación (deterministas).
- construir los DataLoader de entrenamiento/validación/test con un split
  estratificado 85/15 dentro de `training_set`, dejando `test_set` intacto
  para la evaluación final.

A diferencia de `clasificador-perros-gatos` (EfficientNet-B0 con transfer
learning) y de la primera versión de este proyecto, aquí ResNet18 se entrena
**desde cero, sin pesos preentrenados de ImageNet**. Por eso este archivo NO
normaliza las imágenes con las estadísticas de ImageNet (`mean`/`std` de ese
dataset): esa normalización solo tiene sentido cuando se reutilizan pesos
entrenados con ese preprocesamiento exacto. Al entrenar desde cero, la
práctica recomendada es un escalado simple a `[0, 1]`, que es exactamente lo
que hace `transforms.ToTensor()` por sí solo (divide cada píxel entre 255),
sin `transforms.Normalize` adicional. El aumento de datos (flip, rotación,
jitter de color) se mantiene igual —de hecho es más importante todavía aquí,
porque sin el "prior" aprendido de ImageNet el modelo es más propenso a
sobreajustar un dataset de ~10,000 imágenes.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Tamaño de entrada estándar para arquitecturas tipo ResNet/EfficientNet
# entrenadas sobre imágenes de ~224x224. Se mantiene igual al resto de
# proyectos de esta comparación (no es un valor ligado a pesos preentrenados,
# sino simplemente el tamaño de entrada de la red).
TAMANO_REDIMENSION = 256
TAMANO_RECORTE = 224

# Orden alfabético de las subcarpetas del dataset ("cats" antes que "dogs"),
# igual al criterio que usa `torchvision.datasets.ImageFolder` por defecto.
NOMBRES_CLASES = ("gato", "perro")


@dataclass(frozen=True)
class Ejemplo:
    """Un par (ruta de imagen, etiqueta) ya validado."""

    ruta: Path
    etiqueta: int


def _es_imagen_valida(ruta: Path) -> bool:
    """Verifica que el archivo sea una imagen legible con Pillow.

    `Image.verify()` detecta encabezados corruptos o archivos truncados sin
    decodificar el contenido completo (es rápido). También descarta archivos
    que directamente no son imágenes, como los `_DS_Store` que trae el
    dataset de Kaggle.
    """
    try:
        with Image.open(ruta) as imagen:
            imagen.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def validar_y_limpiar_directorio(directorio_clase: Path) -> tuple[list[Path], list[Path]]:
    """Separa imágenes válidas de inválidas dentro de un directorio de una clase.

    Las inválidas se mueven a una subcarpeta `_invalidas` en vez de borrarse,
    para no destruir datos por error y dejar rastro de qué quedó excluido.

    Devuelve (rutas_validas, rutas_movidas_a_invalidas).
    """
    rutas_validas: list[Path] = []
    rutas_invalidas: list[Path] = []
    carpeta_invalidas = directorio_clase / "_invalidas"

    for ruta in sorted(directorio_clase.iterdir()):
        if ruta.is_dir():
            continue
        if _es_imagen_valida(ruta):
            rutas_validas.append(ruta)
        else:
            carpeta_invalidas.mkdir(exist_ok=True)
            destino = carpeta_invalidas / ruta.name
            shutil.move(str(ruta), str(destino))
            rutas_invalidas.append(destino)

    return rutas_validas, rutas_invalidas


def construir_lista_de_ejemplos(directorio_raiz: Path) -> list[Ejemplo]:
    """Valida y lista todos los ejemplos (ruta, etiqueta) bajo `directorio_raiz`.

    Espera la estructura `directorio_raiz/{cats,dogs}/*.jpg`, tal como viene
    el dataset `tongpython/cat-and-dog` de Kaggle dentro de
    `training_set/training_set` y `test_set/test_set`.
    """
    subcarpetas_por_clase = {"cats": 0, "dogs": 1}
    ejemplos: list[Ejemplo] = []

    for nombre_subcarpeta, etiqueta in subcarpetas_por_clase.items():
        directorio_clase = directorio_raiz / nombre_subcarpeta
        if not directorio_clase.is_dir():
            raise FileNotFoundError(
                f"No se encontró la carpeta esperada '{directorio_clase}'. "
                "Verifica que el dataset se haya descargado y descomprimido correctamente."
            )
        rutas_validas, rutas_invalidas = validar_y_limpiar_directorio(directorio_clase)
        if rutas_invalidas:
            nombres = [ruta.name for ruta in rutas_invalidas]
            print(
                f"[dataset] {len(rutas_invalidas)} archivo(s) inválido(s) movidos "
                f"a '{directorio_clase / '_invalidas'}': {nombres}"
            )
        ejemplos.extend(Ejemplo(ruta=ruta, etiqueta=etiqueta) for ruta in rutas_validas)

    return ejemplos


class DatasetPerrosGatos(Dataset):
    """Dataset de PyTorch que carga imágenes de disco bajo demanda (no en RAM)."""

    def __init__(self, ejemplos: list[Ejemplo], transformacion: transforms.Compose):
        self.ejemplos = ejemplos
        self.transformacion = transformacion

    def __len__(self) -> int:
        return len(self.ejemplos)

    def __getitem__(self, indice: int):
        ejemplo = self.ejemplos[indice]
        with Image.open(ejemplo.ruta) as imagen:
            imagen_rgb = imagen.convert("RGB")
            tensor_imagen = self.transformacion(imagen_rgb)
        # BCEWithLogitsLoss espera la etiqueta como float (no como entero long).
        return tensor_imagen, float(ejemplo.etiqueta)


def crear_transformacion_entrenamiento() -> transforms.Compose:
    """Transforms con aumento de datos, exclusivos del set de entrenamiento.

    El aumento (flip horizontal, rotación leve, jitter de color) ayuda a que
    el modelo generalice en vez de memorizar detalles específicos de cada
    foto de entrenamiento. Nunca se aplica a validación/test: ahí se necesita
    medir desempeño sobre datos representativos del uso real, sin distorsión
    artificial. Al entrenar desde cero (sin el "prior" de ImageNet), el
    aumento de datos es aún más importante para evitar sobreajuste.

    `transforms.ToTensor()` ya escala los píxeles de `[0, 255]` a `[0, 1]`
    (los divide entre 255); no se aplica ninguna normalización adicional con
    estadísticas de ImageNet porque el modelo no usa pesos preentrenados.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(TAMANO_RECORTE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
        ]
    )


def crear_transformacion_evaluacion() -> transforms.Compose:
    """Transforms deterministas (sin aumento) para validación, test e inferencia.

    Resize a 256, recorte central a 224 y escalado a `[0, 1]` vía
    `transforms.ToTensor()` (sin normalización de ImageNet, por la misma
    razón que en `crear_transformacion_entrenamiento`).
    """
    return transforms.Compose(
        [
            transforms.Resize(TAMANO_REDIMENSION),
            transforms.CenterCrop(TAMANO_RECORTE),
            transforms.ToTensor(),
        ]
    )


def crear_dataloaders(
    directorio_datos: Path,
    tamano_lote: int = 32,
    proporcion_validacion: float = 0.15,
    semilla: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Arma los DataLoader de entrenamiento, validación y test.

    El split train/val es estratificado (misma proporción de perros y gatos
    en ambos subconjuntos) y se calcula únicamente a partir de
    `training_set`; `test_set` se mantiene completamente aparte y solo se usa
    en la evaluación final, nunca para decisiones de diseño o de hiperparámetros.

    `num_workers=0` por defecto: en Windows, `DataLoader` usa el modo
    `spawn` para crear procesos hijos, lo que obliga a cada worker a
    reimportar PyTorch/SciPy/scikit-learn desde cero. En esta máquina eso
    agotó el archivo de paginación y los workers fallaban con
    `ImportError: DLL load failed`. Con `num_workers=0` la carga de datos
    ocurre en el proceso principal; es algo más lento, pero el cuello de
    botella real de este entrenamiento es la GPU, no la lectura de disco.
    """
    directorio_entrenamiento = directorio_datos / "training_set" / "training_set"
    directorio_test = directorio_datos / "test_set" / "test_set"

    ejemplos_entrenamiento_completo = construir_lista_de_ejemplos(directorio_entrenamiento)
    ejemplos_test = construir_lista_de_ejemplos(directorio_test)

    etiquetas = [ejemplo.etiqueta for ejemplo in ejemplos_entrenamiento_completo]
    ejemplos_train, ejemplos_val = train_test_split(
        ejemplos_entrenamiento_completo,
        test_size=proporcion_validacion,
        random_state=semilla,
        stratify=etiquetas,
    )

    dataset_train = DatasetPerrosGatos(ejemplos_train, crear_transformacion_entrenamiento())
    dataset_val = DatasetPerrosGatos(ejemplos_val, crear_transformacion_evaluacion())
    dataset_test = DatasetPerrosGatos(ejemplos_test, crear_transformacion_evaluacion())

    cargador_train = DataLoader(
        dataset_train,
        batch_size=tamano_lote,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    cargador_val = DataLoader(
        dataset_val,
        batch_size=tamano_lote,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    cargador_test = DataLoader(
        dataset_test,
        batch_size=tamano_lote,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return cargador_train, cargador_val, cargador_test


if __name__ == "__main__":
    # Prueba de humo: confirma que el dataset descargado se puede recorrer sin
    # errores, que el split estratificado respeta la proporción pedida (85/15,
    # verificable a mano) y que cada lote tiene la forma esperada por el modelo.
    directorio_proyecto = Path(__file__).resolve().parent.parent
    directorio_datos = directorio_proyecto / "data"

    cargador_train, cargador_val, cargador_test = crear_dataloaders(
        directorio_datos, tamano_lote=16, num_workers=0
    )

    total_train = len(cargador_train.dataset)
    total_val = len(cargador_val.dataset)
    total_test = len(cargador_test.dataset)

    proporcion_val_real = total_val / (total_train + total_val)
    assert abs(proporcion_val_real - 0.15) < 0.01, (
        f"El split de validación debía ser ~15% del set de entrenamiento, "
        f"dio {proporcion_val_real:.4f}"
    )

    lote_imagenes, lote_etiquetas = next(iter(cargador_train))
    assert lote_imagenes.shape[1:] == (3, TAMANO_RECORTE, TAMANO_RECORTE), (
        f"Se esperaba forma (3, {TAMANO_RECORTE}, {TAMANO_RECORTE}), "
        f"dio {tuple(lote_imagenes.shape[1:])}"
    )
    assert set(lote_etiquetas.tolist()).issubset({0.0, 1.0}), (
        "Las etiquetas deben ser 0.0 (gato) o 1.0 (perro) para BCEWithLogitsLoss"
    )

    print(f"Ejemplos de entrenamiento: {total_train}")
    print(f"Ejemplos de validación:    {total_val}")
    print(f"Ejemplos de test:          {total_test}")
    print(f"Proporción real de validación: {proporcion_val_real:.4f}")
    print(f"Forma de un lote de imágenes: {tuple(lote_imagenes.shape)}")
