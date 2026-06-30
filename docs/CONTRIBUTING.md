# Contributing — как внести вклад в ComfyUI-LTX2-MultiGPU

> Это личный (solo) проект **The Angel Studio** под **GNU GPL v3-or-later** (SPDX: `GPL-3.0-or-later`). Вклады приветствуются, но идут через простой GitHub-flow с DCO-подтверждением — ниже всё по шагам.

Документ на русском с EN-терминами — стандарт для наших README/CHANGELOG/SECURITY.

---

## TL;DR — quick-start для опытных contributor'ов

1. Fork → branch `fix/<short-slug>` / `feat/<short-slug>` / `docs/<short-slug>` → **1 фикс = 1 PR**.
2. Commit messages — **Conventional Commits**: `<type>(<scope>): <subject>` (см. ниже). **DCO sign-off обязателен**: `git commit -s` добавляет `Signed-off-by:` trailer.
3. Перед push:
   * `python -m py_compile nodes.py core/*.py` — должен пройти **без warnings**.
   * Smoke-test минимум: `python -c "import sys; sys.path.insert(0, '.'); import nodes; print('OK')"`.
   * Если меняете `core/` — запустите reproduction минимум на **одной** конфигурации из hardware-matrix ниже.
4. PR description должен содержать 5 пунктов (см. шаблон ниже).
5. Open PR → CI (когда появится) + manual review от maintainer'а → squash-merge с вашим commit'ом + DCO-trail.

---

## License + DCO (DCO обязателен)

Проект под **GPL-3.0-or-later** — это значит:

* **Ваш copyright остаётся у вас.** Вы **не переносите** авторские права — DCO не transfer'ит ownership. Вместо этого вы **сертифицируете**, что имеете право contribute под GPL-3.0-or-later (т.е. либо вы сами автор, либо upstream-код уже под GPL-compatible license). Все downstream-получатели получают ваш contribution на тех же условиях GPL-3.0-or-later, что и весь проект.
* **CLA у нас нет** — это solo project. DCO достаточно.
* **Maintainer (The Angel Studio) — sole copyright holder только собственного кода проекта**, не ваших contribution'ов. Maintainer **не может** самостоятельно relicence'нуть third-party contribution'ы: такие строки кода остаются под GPL-3.0-or-later пока каждый автор отдельно не даст согласия. Если maintainer когда-нибудь решит изменить license проекта, contribution'ы от author'ов кроме maintainer'а потребуют отдельной лицензии совместимости с новым license'ом.
* **DCO sign-off** означает добавление строки `Signed-off-by: Your Name <your@email>` в trailer каждого commit'а. Сделать это в `git`:

  ```bash
  git config user.name  "Your Name"
  git config user.email "your@email"
  git commit -s -m "fix(apply_strategy): clamp verbose_log before downstream call"
  ```

  Сообщение DCO в commit-trail — стандартный Linux Foundation / Git / Kubernetes / Chromium паттерн: <https://developercertificate.org/>. Подтверждая DCO, вы заявляете, что имеете право contribution'а под данным license.
* **DCO-bot** (`probot/dco`) автоматически проверит, что все commit'ы в PR подписаны. Неподписанные commit'ы НЕ merge'нутся автоматически.

Если вы **не** можете или **не хотите** подтвердить DCO — опишите ситуацию в PR-описании, мы обсудим индивидуально (альтернативой может быть перенос вашего кода в форк без merge в upstream).

---

## Code of Conduct — короткая версия

Этот проект придерживается принципов из **[CNCF Code of Conduct](https://github.com/cncf/foundation/blob/main/code-of-conduct.md)** (адаптированный короткий):

* **Будьте доброжелательны.** Особенно к новым участникам — они задают те же вопросы, что и вы год назад.
* **Argumentum ad personam — нет.** Техническая критика кода / design'а — yes. Личные нападки — no.
* **Security-вопросы — в private**, не в issue-трекере. См. [`./SECURITY.md`](./SECURITY.md).

Maintainer (The Angel Studio) оставляет за собой право закрывать thread'ы / issues которые нарушают CoC, с публичным объяснением причины. Эта мера применяется редко и в крайних случаях.

---

## Fork → Branch → PR — стандартный GitHub-flow

Ветвимся **только** от `main`. Никаких `develop` / `release/*` веток в этом проекте нет — это solo codebase, Git Flow overhead не оправдан.

### Branch naming convention

* `fix/<short-slug>` — bug-fix. Например: `fix/oom-on-single-gpu-fallback`.
* `feat/<short-slug>` — new feature или API-расширение. Например: `feat/donor-device-cpu-options`.
* `docs/<short-slug>` — README / CHANGELOG / SECURITY / комментарии. Например: `docs/clarify-warn-on-single-gpu`.
* `refactor/<short-slug>` — internal-only изменение, не меняющее public behavior (структура файлов, rename приватных names'ов, etc.).
* `perf/<short-slug>` — performance-fix.
* `test/<short-slug>` — добавляет test-coverage.
* `chore/<short-slug>` — технический (CI, .gitignore, pyproject bump).

`<short-slug>` — kebab-case, ≤ 5 слов, **конкретно-описательный**. Избегайте generic `fix/issue` / `feat/stuff`. Хороший slug может прочитать reviewer за 2 секунды и понять scope PR.

### «1 фикс = 1 PR» — почему и как

* **Почему**: 1 PR = 1 atomic change. Если в одном PR намешаны fix A + refactor B + feat C, в code review вы будете обсуждать все три, и если C — спорное, A и B застрянут с ним.
* **Как**: если у вас есть **несколько** несвязанных фиксов, делайте несколько веток и несколько PR. Если нужен **большой** рефактор +405/-200, его можно оформить как 1 PR, но лучше squash его в atomic-коммиты в самом PR (потом `Squash and merge` сделает один commit'т в main).

### PR description — минимум 5 пунктов

Шаблон (вставьте в `<!-- PR template -->` ниже при создании PR):

```markdown
## Reproduction

Минимальный Python snippet **или** workflow_api.json показывающий проблему, **до** PR.

## Solution

Что меняет PR в 1–3 предложениях. Ссылка на issue #N если есть.

## Test setup

- Python version: 3.11 / 3.12
- torch version: 2.x.x
- Hardware: T4×2 Kaggle / RTX 4090×2 / A5000+3090 / другое: _______
- OS: Windows 11 / Ubuntu 22.04 / Colab / Kaggle
- Smoke-test result: `python -m py_compile nodes.py core/*.py` → OK
- Optional additional: фактическая VRAM-budget свободная после loading (MB)

## Risk assessment

Backward-compat: API breaks / signature changes / new deps / etc. Если ничего — напишите "no public API change".

## Checklist

- [ ] `git commit -s` (DCO Signed-off-by: есть в каждом commit'е)
- [ ] Branch naming follows convention (`fix/...`, `feat/...`, etc.)
- [ ] Conventional Commits message format
- [ ] py_compile passes на Python 3.10 / 3.11 / 3.12 (если возможно)
- [ ] Reproduction приложен и проверен на hardware
- [ ] Никаких untracked файлов в PR (`.gguf`, `.safetensors`, `__pycache__/`, `outputs/` — игнорятся, но всё равно проверьте)
```

---

## Commit messages — Conventional Commits

Стандарт: **Conventional Commits** 1.0.0 (<https://www.conventionalcommits.org/ru/>). Формат:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### `<type>`

> **CC 1.0.0 не обязывает к конкретному type-list** — это де-факто стандарт; `chore` / `security` / `style` ниже — общепринятые extensions, а не формальные расширения spec. См. <https://www.conventionalcommits.org/ru/> для базового типа-list.

* `feat` — новая user-visible functionality.
* `fix` — bug-fix.
* `docs` — только README / CHANGELOG / SECURITY / комментарии. Не меняет code.
* `refactor` — internal-only изменение без public-API change.
* `perf` — performance-fix (без behavior change).
* `test` — добавляет/правит tests.
* `ci` — GitHub Actions / pre-commit / .gitignore.
* `build` — pyproject.toml / setup.py / requirements.txt / версии зависимостей.
* `chore` — технический misc (LICENSE tooling update, etc.).
* `security` — security-fix (КРОМЕ того что идёт в [`./SECURITY.md`](./SECURITY.md) policy-файл).
* `style` — formatting-only (Black / reorder imports / etc.), не меняет logic.

### `<scope>` (опционально, но рекомендуется)

Конкретный scope в этом проекте:

* `nodes` — `nodes.py` (4 ComfyUI-класса)
* `gguf_split` / `gguf_reader` / `memory_tracker` — отдельные modules в `core/`
* `init` — `__init__.py` (FOLDER_PATHS_OK flag, проверка путей)
* `readme` / `changelog` / `security` / `contributing` — отдельные docs-файлы
* `pyproject` / `ci` — meta-files
* **NO scope** — если изменение cross-cutting (например `feat: add Boosty badge` затрагивает и README, и FUNDING.yml).

### `<subject>`

* **Императивное наклонение** (present tense): "add", "fix", "remove", **НЕ** "added", "fixes", "removed".
* **Lowercase** (исключение — proper nouns: `GPL-3.0`, `GGUF`, `ComfyUI`).
* **Без period в конце.**
* **≤ 72 символов.**
* **Никаких** смайликов / emoji в subject (в body — допустимо, но только в `docs:` scope).

### `<body>` (опционально, для нетривиальных PR)

Wrap на **72 символа**. Объясняет **что и почему**, **не как** (как — в diff'е). Ссылка на issue: `Closes #42` или `Refs #42`.

### `<footer>` (DCO обязателен)

```
Signed-off-by: Your Name <your@email>

Closes #42
```

---

## Code style

### Python formatting

* **Formatting**: следуйте [PEP 8](https://peps.python.org/pep-0008/) + soft-wrap **88 символов** (Black-compatible default). Внутри проекта пока нет `pyproject.toml [tool.black]` / `.pre-commit-config.yaml` / `ruff.toml` — **когда CI приземлит `ruff format` или `black --check`, эти правила станут authoritative**. До CI — manual PEP 8 review от maintainer'а в PR. Если у вас локальное несогласие с правилом, добавляйте `# fmt: skip` рядом со строкой-исключением и комментируйте **почему**.
* **Type hints обязательны** на всех `def`-уровнях (`def load_gguf(path: str) -> dict[str, torch.Tensor]:`) и class-methods. На private helpers (`_install_cross_device_hook`) — тоже.
* **`from __future__ import annotations`** — обязательно в начале каждого `.py`, чтобы hints работали на Python 3.10.
* **NO bare `except:` и не `except Exception:` если можно сузить.** Используйте конкретно: `except (RuntimeError, AssertionError):`, etc. Если broad-catch неизбежен — комментируйте почему и ставьте `# noqa: BLE001`.
* **Imports order** (Black/Ruff compatible): stdlib → third-party → local (`from core import gguf_split`). `from __future__ import annotations` всегда первая строка в файле.

### Russian comments — стиль

* Технические комментарии и docstrings — **на русском** для объяснения "почему" или "как".
* EN в коде — для things которые циркулируют в международном OSS-пространстве: API names, error messages, log lines (поэтому WARN — на английском: пользователи ищут по stack-traces).
* Пример допустимого стиля:

  ```python
  def _cuda_donor_choices(include_cpu: bool = False) -> list[str]:
      """FIX LOW #4+#6+#LOW_cpu_machine: динамический donor_device-список.

      Использует torch.cuda.device_count() если torch доступен И CUDA активен.
      На CPU-only машинах / torch=None fallback **всё равно** показывает cuda:0/cuda:1
      как conservative defaults — это сохраняет контракт исходного
      ``_DONOR_DEVICE_CHOICES_DIT = ("auto", "cuda:0", "cuda:1")``.
      """
  ```

### Naming conventions

* snake_case для функций и переменных.
* PascalCase для ComfyUI node-классов (как требует ComfyUI registry).
* `_private` prefix для внутренних helpers.
* Dunder `__init__`, `__post_init__` — по PEP 8.

### Forbidden patterns

* **No mutating `self.list` directly** в ComfyUI node-class methods — ComfyUI re-uses instances между workflow'ами. Use `self._state_cache` patterns.
* **No global `torch.cuda.empty_cache()` на уровне модуля** — это дорого и неожиданно для других nodes. Если нужен empty_cache, делайте в try-finally scope.
* **No hardcoded paths** (типа `C:/Users/the_angel/...`). Используйте `folder_paths.get_input_directory()` (ComfyUI convention).

---

## Что нужно приложить к PR (reproduce + py_compile + hardware)

### Reproduction (для `fix:` PR)

Минимальный Python snippet **ИЛИ** `workflow_api.json` для ComfyUI, воспроизводящий **проблему ДО PR**. После PR — тот же snippet должен работать корректно.

Пример минимального snippet:

```python
import sys, os
sys.path.insert(0, '.')   # hack for ad-hoc testing

from nodes import LTX2_MultiGPU_MemoryDiagnostics
node = LTX2_MultiGPU_MemoryDiagnostics()
report = node.diagnose(verbose=True)[0]
print("VRAM free:", report["free_mb"])
```

### py_compile (минимум)

```bash
python -m py_compile nodes.py core/gguf_split.py core/gguf_reader.py core/memory_tracker.py
```

Должен пройти **без output** и **exit-code 0**. Если есть warnings — fix'ите, **no noqa-комментариев** для compile-warnings.

### Hardware matrix (для `feat:` / `fix:` с model-loading effects)

Опишите **где** тестировали. Если у вас доступ только к одной конфигурации — укажите её явно. Maintainer покроет остальные конфигурации перед release.

| Hardware | CUDA версия | Driver | Где тестировалось |
|:---------|:-----------:|:------:|:-----------------:|
| T4 × 2 (Kaggle) | 12.x | 535.x | Kaggle GPU 2x |
| RTX 4090 × 2 | 12.x | 555.x | Local workstation |
| A5000 + 3090 (asymmetric) | 12.x | 555.x | Local workstation |
| A100 × 1 (Colab) | 12.x | 535.x | Colab Pro |
| CPU-only (no CUDA) | — | — | Local workstation |

Если добавили **новое hardware** в matrix — это хорошо; maintainer калибрует cross-check перед merge'ом.

---

## Issue reporting — куда писать

* **Bug reports / functional issues** → [GitHub Issues](https://github.com/THE-ANGEL-AI/ComfyUI-LTX2-MultiGPU/issues) (label `bug`).
* **Feature requests / UX-questions** → [GitHub Issues](https://github.com/THE-ANGEL-AI/ComfyUI-LTX2-MultiGPU/issues) (label `enhancement`) или [GitHub Discussions](https://github.com/THE-ANGEL-AI/ComfyUI-LTX2-MultiGPU/discussions).
* **Security vulnerabilities** → **НЕ в issue-tracker**, а через [`./SECURITY.md`](./SECURITY.md) channels (GitHub Private Security Advisory `https://github.com/THE-ANGEL-AI/ComfyUI-LTX2-MultiGPU/security/advisories/new` или email `gi.the.angel@gmail.com` с `[SECURITY ...]` subject).

---

## Out of scope — что мы НЕ принимаем

* **Изменения, несовместимые с GPL-3.0-or-later** (попытки relicense в MIT / Apache / etc.) — обсуждаются только если инициирует maintainer.
* **Pure formatting-only PR** (rename / reorder imports / no functional change). Refactor-only PR'ы приветствуются, **но сначала откройте issue** чтобы согласовать scope и размер с maintainer'ом. Когда `ruff format` или `black --check` приземлится в CI, pure-formatting PR станет **автоматически проверяться formatter'ом** и merge'иться без manual drift-checks — сегодня это делается глазами. Для behavior-preserving внутренних рефакторингов используйте `refactor:` scope, для formatting-only — `style:` scope.
* **Зависимости, увеличивающие install size >100 МБ** (типа `transformers==4.x` полный пакет, когда нужен только `tokenizer`) — обсуждайте заранее в Discussions.
* **Лишние "defensive" try/except** без конкретного recovery plan — если вы не знаете, что делать в `except`, лучше **пробросить error наверх** и пусть ComfyUI покажет stack-trace, чем silent fallback.
* **PR без DCO sign-off** в commit. Закрывается автоматически через DCO-bot.

---

## Версия этого документа

* **v1.0** — initial publication (2026-06-30), синхронно с release v0.2.0 / [`./SECURITY.md`](./SECURITY.md) v1.0 / [`./CHANGELOG.md`](./CHANGELOG.md) v0.2.0.

Maintainer: **The Angel Studio** · Contact: `gi.the.angel@gmail.com` · Repo: `https://github.com/THE-ANGEL-AI/ComfyUI-LTX2-MultiGPU`

`CONTRIBUTING.md` распространяется под **GPL-3.0-or-later** (как и код проекта).
