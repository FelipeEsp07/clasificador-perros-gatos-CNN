# Handoff: control-versiones (git-devp)
Última actualización: 2026-08-21

## Resumen
Se inicializó el repositorio git de `clasificador-perros-gatos-CNN` (no existía
control de versiones previo) y se subió a GitHub como repositorio nuevo,
público, en la cuenta `FelipeEsp07`. El historial se organizó en 5 ramas de
feature, cada una fusionada a `main` mediante pull request (con commit de
merge, sin squash), para dejar evidencia legible de la estructura del
desarrollo en vez de un único commit gigante de "subida inicial".

URL del repositorio: https://github.com/FelipeEsp07/clasificador-perros-gatos-CNN

## Verificación previa a la fusión (checklist del rol)
- Leí `.claude/handoffs/tester-qa.md` (único handoff existente en este
  proyecto; no había `desarrollador-backend.md` ni `desarrollador-frontend.md`
  — el propio QA lo documentó como una limitación de este proyecto concreto,
  validando el contrato contra `README.md` y el código fuente directamente).
- QA no reportó defectos bloqueantes ni de severidad alta sobre el código en
  sí. El único hallazgo de severidad ALTA (#1, servidor Streamlit huérfano en
  el puerto 8503) es un problema de higiene de procesos de la sesión de
  pruebas, no un defecto de la aplicación — QA confirmó que ya lo resolvió
  (mató ambos procesos, puerto liberado) y el veredicto final de QA es
  explícito: "funciona correctamente y no tiene bloqueantes funcionales".
- No había contratos de backend/frontend que verificar entre sí (no existían
  esos handoffs), así que no aplica el punto de "discrepancia de contratos".
- Sin secretos ni credenciales en el código (verificado con búsqueda de
  patrones `api_key|secret|password|token` sobre todo el árbol versionable:
  sin coincidencias).
- Verifiqué con `git add -A --dry-run` antes de cualquier commit que
  `venv/`, `data/`, `wandb/`, `__pycache__/` y `*.log` quedaran excluidos —
  el `.gitignore` ya venía correcto, no fue necesario ajustarlo.

## Estructura de ramas y qué contiene cada una

Todas las ramas se crearon desde `main`, se subieron a `origin`, y se
fusionaron con `gh pr create` + `gh pr merge --merge` (commit de merge, sin
squash, ramas conservadas sin borrar para que la estructura sea visible en
GitHub).

| Rama | PR | Contenido |
|---|---|---|
| `setup-proyecto` | #1 | `.gitignore`, `requirements.txt`, `README.md` |
| `pipeline-datos` | #2 | `src/dataset.py` |
| `modelo-y-entrenamiento` | #3 | `src/model.py`, `src/train.py`, `src/evaluate.py` |
| `app-streamlit` | #4 | `app/streamlit_app.py` |
| `modelo-entrenado` | #5 | `models/best_model.pth` (93.38% accuracy en test, ~44.7MB), `models/matriz_confusion.png` |
| `docs/handoffs-proceso` | #6 | `.claude/handoffs/tester-qa.md`, `.claude/handoffs/control-versiones.md` (este archivo) |

`main` arranca con un commit vacío `chore: commit inicial del proyecto`
(requisito de higiene del rol git-devp para cualquier repositorio nuevo, para
que operaciones como `git worktree add` puedan resolver HEAD si en el futuro
otro agente trabaja aislado sobre este repo) y luego incorpora cada rama
mediante su commit de merge correspondiente.

## Decisiones y supuestos
- El checkpoint `models/best_model.pth` se subió a propósito (decisión
  explícita del usuario, documentada también en el mensaje de commit de la
  rama `modelo-entrenado`): ~44.7MB, muy por debajo del límite de 100MB de
  GitHub, no requiere Git LFS.
- Los handoffs de proceso (`tester-qa.md`, `control-versiones.md`) se
  agruparon en su propia rama `docs/handoffs-proceso` en vez de incluirlos en
  `setup-proyecto`, porque documentan el proceso de desarrollo completo
  (incluyendo QA sobre la app ya terminada), no la estructura inicial del
  proyecto.
- Se usó `gh pr merge --merge` (commit de merge explícito) en vez de squash o
  rebase, para que el grafo de git conserve la evidencia de cada rama de
  feature como la pidió el usuario ("que quede evidencia de las ramas en el
  historial").
- Las ramas de feature no se borraron tras el merge, para que la lista de
  ramas en GitHub siga mostrando la organización del trabajo.

## Pendientes (TODO) heredados de QA, no resueltos en esta sesión
Estos quedan documentados aquí para trazabilidad, pero no se resolvieron
como parte de la tarea de subir el repositorio (son cambios de código, fuera
del alcance de control de versiones):
- Backend: aplicar `weights_only=True` en `torch.load` de
  `app/streamlit_app.py` (línea ~41).
- Frontend: reemplazar `use_container_width=True` por `width='stretch'` en
  `app/streamlit_app.py` (línea ~83).
- Agregar suite de tests automatizados (`pytest`) para `predecir_clase()` y
  el manejo de `UnidentifiedImageError` — no existía ninguna suite de tests
  en el proyecto al momento de esta subida.

## Estado final del repositorio
- Público: sí.
- Rama por defecto: `main`, con las 5 ramas de feature + la de handoffs ya
  fusionadas (6 PRs en total, todos mergeados).
- Sin commits vacíos salvo el inicial (obligatorio por convención del rol,
  documentado arriba) y sin mensajes genéricos tipo "wip" o "cambios varios".

## Actualización 2026-08-22 — limpieza de referencias a EfficientNet-B0

El usuario pidió eliminar toda mención al proyecto hermano
`clasificador-perros-gatos` (variante EfficientNet-B0 + transfer learning),
que nunca se subió a GitHub y no debe quedar documentado en este repo
público, ni en su "propósito comparativo de tres puntas".

### Verificación previa (checklist del rol)
- Leí `.claude/handoffs/tester-qa.md` (único handoff funcional existente en
  este proyecto además de este archivo). Sin defectos bloqueantes: el único
  hallazgo de severidad ALTA es ambiental (proceso Streamlit huérfano) y ya
  quedó resuelto en la sesión de QA; los dos hallazgos restantes son de
  severidad BAJA y no bloquean esta limpieza de documentación.
- No hay handoffs de `desarrollador-backend.md` ni `desarrollador-frontend.md`
  en este proyecto, así que no aplica la verificación de contratos cruzados.
- Revisé el diff completo (`git diff`) de los 6 archivos modificados
  localmente antes de commitear: confirmé que **todos los cambios son
  docstrings, comentarios y contenido de `README.md`**, sin ninguna línea de
  lógica de negocio alterada en `src/` ni en `app/streamlit_app.py`.

### Qué se hizo
1. **Commit directo a `main`** (sin rama/PR — limpieza de documentación
   menor, no ameritaba el patrón de PR usado para features): commit
   `04f57ac`, mensaje `docs: eliminar referencias comparativas al proyecto
   EfficientNet-B0`. Pusheado a `origin/main` sin conflictos
   (`f628521..04f57ac`).
   - Archivos: `README.md` (se quitó la sección "Propósito comparativo" y la
     tabla de diferencias frente a EfficientNet-B0), `src/model.py`,
     `src/train.py`, `src/dataset.py`, `src/evaluate.py`,
     `app/streamlit_app.py` (docstrings/comentarios reescritos sin mención
     al proyecto hermano ni a EfficientNet-B0).
2. **Descripciones de Pull Requests editadas** (`gh pr edit <n> --body ...`),
   sin tocar el historial de commits ya mergeado:
   - PR #1 (`setup-proyecto`): se quitó "vs. transfer learning" del propósito
     del experimento, reformulado como "entrenamiento de ResNet18 desde cero
     (sin pesos preentrenados de ImageNet)". Resto de la descripción
     (contenido del PR) se mantuvo igual.
   - PR #3 (`modelo-y-entrenamiento`): se quitó "Es la contraparte 'entrenada
     desde cero' del experimento comparativo... frente a una variante con
     transfer learning", reformulado sin mencionar comparación ni proyecto
     hermano.
   - PR #2, #4, #5, #6: revisados, no mencionaban EfficientNet-B0 ni el
     propósito comparativo — no requirieron edición.
3. **Descripción del repositorio** (`gh repo edit --description`): se quitó
   "Proyecto comparativo educativo" → queda "Proyecto educativo con PyTorch y
   Streamlit".
4. **Nombres de rama**: revisados (`app-streamlit`, `docs/handoffs-proceso`,
   `main`, `modelo-entrenado`, `modelo-y-entrenamiento`, `pipeline-datos`,
   `setup-proyecto`) — ninguno menciona "efficientnet" ni "b0", no requirió
   cambios.
5. **No se tocó el historial de commits** ya mergeado (mensajes de los 6 PRs
   originales quedan intactos como registro histórico), conforme a la
   instrucción explícita del usuario de no reescribir historial.

### Verificación final
- `grep -ri "efficientnet|tres puntas|comparativ"` sobre el árbol local del
  repo: sin coincidencias.
- Búsqueda del mismo patrón sobre los `body` de los 6 PRs vía `gh pr list
  --json body`: sin coincidencias.
- Descripción del repo verificada vía `gh repo view --json description`: sin
  mención a EfficientNet-B0 ni comparación.

URL final del repositorio:
https://github.com/FelipeEsp07/clasificador-perros-gatos-CNN
