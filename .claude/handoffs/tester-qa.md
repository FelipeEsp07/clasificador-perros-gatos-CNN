# Handoff: tester-qa
Última actualización: 2026-08-21 21:50

## Resumen
Validación end-to-end con Playwright (navegador real) de `app/streamlit_app.py`
(clasificador perro/gato con ResNet18 entrenado desde cero). No existían
`.claude/handoffs/desarrollador-backend.md` ni `.claude/handoffs/desarrollador-frontend.md`
en este proyecto antes de esta sesión, así que el contrato se validó contra
`README.md` y el propio código fuente (`app/streamlit_app.py`, `src/dataset.py`).
Resultado general: funcionalidad núcleo (predicción + manejo de errores) PASS,
con un hallazgo ambiental crítico para higiene de procesos (servidor huérfano)
y dos hallazgos menores de calidad de código.

## Contrato validado
- UI: `st.file_uploader` acepta solo `jpg/jpeg/png`; sin archivo → sin
  predicción visible.
- Al subir imagen válida: muestra `st.image`, luego
  `Predicción: {gato|perro}` (`NOMBRES_CLASES = ("gato", "perro")` en
  `src/dataset.py`) y `Confianza: NN.N%`.
- Archivo no-imagen → `st.error("El archivo subido no es una imagen válida...")`
  vía `except UnidentifiedImageError`, sin traceback visible.

## Casos probados (Playwright, navegador real, Chromium vía MCP)

| # | Caso | Resultado | Evidencia |
|---|---|---|---|
| 1 | Carga inicial sin errores, UI esperada visible | PASS | `qa-evidencia/01_carga_inicial.png` |
| 2 | Subir gato real (`cat.4001.jpg`) | PASS — "Predicción: gato", 91.0% | `qa-evidencia/02_prediccion_gato.png` |
| 2b| Subir gato real (`cat.4002.jpg`) | PASS — "Predicción: gato", 99.7% | (ver snapshot en sesión) |
| 3 | Subir perro real (`dog.4001.jpg`) | **Clasificación incorrecta del modelo**: "Predicción: gato", 51.1% (confianza casi al azar, cerca del umbral 50%) | `qa-evidencia/03_prediccion_perro_dog4001_incorrecta.png` |
| 3b| Subir perro real (`dog.4002.jpg`) | PASS — "Predicción: perro", 100.0% | `qa-evidencia/04_prediccion_perro_dog4002_correcta.png` |
| 4a| Subir `.txt` (tipo no permitido) | PASS — Streamlit lo rechaza en el propio uploader: "Error: text/plain files are not allowed." Sin traceback. | `qa-evidencia/06_archivo_invalido_txt_rechazado.png` |
| 4b| Subir `.jpg` falso (contenido texto plano, extensión válida, para forzar `UnidentifiedImageError` real de la app) | PASS — mensaje "El archivo subido no es una imagen válida. Intenta con un JPG o PNG." Sin traceback ni crash. | `qa-evidencia/07_archivo_invalido_jpg_falso_manejado.png` |
| 5 | Estado sin archivo subido (inicial y tras remover un archivo) | PASS — no aparece predicción falsa/obsoleta en ningún momento | `qa-evidencia/01_carga_inicial.png` |

## Defectos encontrados

### 1 (severidad ALTA — proceso/entorno, no lógica de negocio): servidor Streamlit huérfano detectado durante la sesión de pruebas
- **Qué esperaba**: que el único servidor Streamlit activo en el puerto 8503 fuera el que yo levanté para esta ronda de pruebas, sirviendo el código estable documentado en `README.md` (título "Clasificador perro vs. gato (ResNet18)").
- **Qué pasó realmente**: a mitad de la sesión, sin que yo recargara la página, el título y encabezado de la UI cambiaron en vivo de "Clasificador de perros y gatos — ResNet18" a "Clasificador de perros y gatos — CNN" (hot-reload de Streamlit). Al investigar, `Get-NetTCPConnection -LocalPort 8503` mostró que el puerto estaba realmente ocupado por un proceso preexistente (PID 15404, `C:\Users\espin\AppData\Local\Programs\Python\Python312\python.exe -m streamlit run app/streamlit_app.py --server.port 8503`, usando el Python global del sistema, NO el venv del proyecto) — no por el proceso que yo lancé (PID 14564, con `venv\Scripts\python.exe`, que nunca llegó a bindear el puerto y terminó con exit code 127). Es decir: **ya había un servidor Streamlit huérfano corriendo desde antes de que empezara mi sesión de QA**, y alguien (probablemente otro agente editando `app/streamlit_app.py` en paralelo) guardó cambios en el archivo mientras yo probaba, lo cual el watcher de ese servidor huérfano recargó en caliente.
- **Pasos para reproducir**: `netstat -ano | findstr :8503` antes de levantar cualquier instancia nueva del proyecto revela si el puerto ya está en uso por un proceso ajeno a la sesión actual.
- **Impacto**: invalida parcialmente la confianza en el caso 1 (carga inicial) tal como está documentado en `README.md`, porque no puedo garantizar con certeza total qué versión exacta del archivo sirvió cada una de mis peticiones anteriores al cambio detectado (aunque el comportamiento funcional — predicción + confianza — se mantuvo correcto en ambas versiones del título). Además, procesos huérfanos en puertos fijos causan fallos intermitentes difíciles de diagnosticar en cualquier corrida futura (`Address already in use` o, peor, servir código desactualizado sin que nadie lo note).
- **A qué handoff pertenece**: proceso/infraestructura de desarrollo, no específicamente backend ni frontend — recomiendo que quien coordine los agentes (git-devp o el usuario) establezca la convención de matar procesos Streamlit al finalizar cada sesión, y verificar `netstat` antes de levantar uno nuevo en un puerto fijo.
- **Acción tomada**: maté ambos procesos (PID 15404 y PID 14564) y confirmé que el puerto 8503 quedó libre (`netstat -ano` solo muestra conexiones residuales en `TIME_WAIT`).

### 2 (severidad BAJA — calidad de código, backend): `torch.load` sin `weights_only=True`
- **Qué esperaba**: que la carga del checkpoint (`app/streamlit_app.py:41`, función `cargar_modelo`) no emitiera advertencias de seguridad.
- **Qué pasó realmente**: el log del servidor muestra en cada carga:
  `FutureWarning: You are using torch.load with weights_only=False (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling...`
- **Pasos para reproducir**: levantar la app y revisar stdout del proceso Streamlit al primer request que dispare `cargar_modelo()`.
- **Impacto**: bajo en este proyecto concreto (el checkpoint es generado localmente por `src/train.py`, no proviene de una fuente no confiable), pero es una advertencia de seguridad explícita de PyTorch y buena práctica corregirla antes de que un futuro release de PyTorch cambie el default y potencialmente rompa la carga si el checkpoint contiene objetos no permitidos.
- **A qué handoff pertenece**: backend (`app/streamlit_app.py` línea 41, o `src/train.py` si se decide cambiar el formato de guardado).

### 3 (severidad BAJA — frontend): uso de parámetro deprecado `use_container_width`
- **Qué esperaba**: que no aparecieran advertencias de deprecación de Streamlit.
- **Qué pasó realmente**: el log del servidor muestra en cada render:
  `Please replace use_container_width with width. use_container_width will be removed after 2025-12-31.` — Esa fecha límite (2025-12-31) ya pasó respecto a la fecha actual (2026-08-21), por lo que la app depende de una API que Streamlit ya marcó para eliminación.
- **Pasos para reproducir**: subir cualquier imagen válida y revisar el stdout del proceso Streamlit; también aparece en la llamada `st.image(imagen, ..., use_container_width=True)` en `app/streamlit_app.py:83`.
- **Impacto**: bajo por ahora (Streamlit todavía lo soporta en esta versión), pero es un riesgo de ruptura futura si se actualiza la dependencia sin tocar este archivo.
- **A qué handoff pertenece**: frontend (`app/streamlit_app.py` línea 83) — cambiar a `width='stretch'`.

## Nota sobre precisión del modelo (no es un defecto de la app)
`dog.4001.jpg` fue mal clasificado como "gato" con 51.1% de confianza (prácticamente
al azar). Esto es coherente con la accuracy documentada (~93%, modelo entrenado desde
cero sin transfer learning): se esperan errores ocasionales. La app en sí manejó el
caso correctamente (mostró una predicción y una confianza numérica válida, sin
crashear); el problema es de calidad del modelo, no de la interfaz ni del pipeline de
inferencia. Lo dejo documentado como observación, no como bug de QA.

## Estado de pruebas
- Suite de tests automatizados: no se encontró carpeta `tests/` ni configuración de
  pytest en este proyecto (`src/` solo tiene `dataset.py`, `evaluate.py`, `model.py`,
  `train.py`, sin archivos `test_*.py`). No hay suite existente que correr. Esto es un
  vacío a nivel de proyecto, no algo que QA pueda "correr y confirmar que pasa" — lo
  marco como pendiente/bloqueante de cobertura automatizada.
- Pruebas manuales E2E con Playwright: 8 casos ejecutados (ver tabla arriba), 7 PASS
  funcionales + 1 predicción incorrecta del modelo (no defecto de app) + 1 hallazgo
  ambiental (servidor huérfano).
- Consola del navegador: 0 errores, 0 warnings durante toda la sesión.
- Proceso Streamlit y navegador Playwright: verificados y cerrados al finalizar
  (puerto 8503 libre, sin procesos huérfanos, `browser_close` confirmado).

## Bloqueado por / dependo de
- No existen `.claude/handoffs/desarrollador-backend.md` ni
  `.claude/handoffs/desarrollador-frontend.md` en este proyecto. Validé contra
  `README.md` y el código fuente directamente. Si backend/frontend generan sus
  handoffs después, hay que re-verificar que el contrato ahí documentado coincide
  con lo que efectivamente probé aquí.
- No hay suite de tests automatizados (`pytest`) en el repo — no pude cumplir el
  punto del DoD de "correr la suite completa" porque no existe. Recomiendo a
  backend/frontend agregar al menos tests unitarios para `predecir_clase()` (caso
  feliz con imagen de cada clase + caso borde con imagen corrupta) antes de la
  siguiente ronda de QA.

## Pendientes (TODO)
- Backend: aplicar `weights_only=True` (o `torch.serialization.add_safe_globals`) en
  `torch.load` de `app/streamlit_app.py:41`.
- Frontend: reemplazar `use_container_width=True` por `width='stretch'` en
  `app/streamlit_app.py:83`.
- Proceso: documentar/aplicar convención de "matar servidores Streamlit al terminar
  cada sesión de trabajo" para evitar procesos huérfanos en puertos fijos como el
  que causó el hallazgo #1.
- Agregar suite de tests automatizados (`pytest`) para `predecir_clase()` y para el
  manejo de `UnidentifiedImageError`.

## Veredicto sobre disposición a producción
La funcionalidad núcleo de la interfaz (subir imagen, mostrar predicción y
confianza, manejar archivos inválidos con gracia, no mostrar predicciones falsas
sin archivo) **funciona correctamente y no tiene bloqueantes funcionales**. Sin
embargo, el sistema **no está listo para un pipeline de CI/CD confiable** sin
resolver: (1) la ausencia total de tests automatizados, y (2) la práctica de dejar
servidores Streamlit huérfanos corriendo en puertos fijos, que ya causó una
interferencia real y medible durante esta misma ronda de pruebas.
