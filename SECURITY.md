# Security Policy — Политика безопасности

Этот документ описывает, как **сообщать** об уязвимостях в проекте **ComfyUI-LTX2-MultiGPU** и что ожидать от нас в ответ.

Проект распространяется под **GNU GPL v3-or-later** (SPDX: `GPL-3.0-or-later`); sole copyright holder = **The Angel Studio**.

> Документ на русском с техническими EN-терминами — стандарт для наших README/CHANGELOG.

---

## Поддерживаемые версии (Supported Versions)

| Версия | Поддержка | Дата релиза | Git SHA |
|:-------|:---------:|:-----------:|:-------:|
| **v0.2.0** (current) | ✅ **Active** — security fixes выпускаются сюда | 2026-06-30 | тег `v0.2.0` создаётся при announcement |
| **v0.1.0** (deprecated) | ⚠️ **Maintenance only** — приём баг-репортов, backport по нашему усмотрению | 2026-06-29 | `f8d553d` (initial public) |
| `pre-0.1.0` (dev) | ❌ **EOL** — только upstream `main` | — | `d8e0a7d`, `366c574` |

**Правило**: security-patch-релизы (например `v0.2.1`) делаются **только для `v0.2.0`**. На `v0.1.0` мы можем портировать fix в `v0.2.0` без backport-релиза. Минимальный поддерживаемый release = текущая стабильная major.minor.

---

## Как сообщить об уязвимости (Reporting a Vulnerability)

### Каналы — в порядке приоритета

#### 1. GitHub Private Security Advisory ⭐ рекомендуется

URL: `https://github.com/THE-ANGEL-AI/ComfyUI-LTX2-MultiGPU/security/advisories/new`

* Заполните форму **"Report a vulnerability"** — GitHub сохранит переписку в private до того, как мы выберем момент публичного disclosure.
* Используйте если у вас есть **CVE-class уязвимость** с публичным CVE id, или вы хотите **coordinated disclosure** со стандартным 90-дневным timeline.
* Лучший вариант для багов с потенциальным PoC — exploit не утечёт до patch'а.

#### 2. Email: `gi.the.angel@gmail.com`

* В subject обязательно: `[SECURITY ComfyUI-LTX2-MultiGPU] <one-line summary>`
* PGP-ключ пока **не публикуем** — для проекта размера v0.2.0 это overkill. Если вам нужен NDA → укажите в первом письме; обсудим канал.
* Используйте если GH Private Advisory неудобен или ваш репорт не требует coordinated disclosure.

#### 3. GitHub Public Issue или Discussion — **только для НЕ-чувствительных багов**

* ❌ **НЕ используйте** для security-уязвимостей — public issue видна сразу, PoC может быть прочитан до patch'а, и у вас не будет контроля над раскрытием.
* ✅ Используйте для функциональных багов, performance issues, UX-предложений, doc-fixes — там `exploit-risk = 0`.

### Что включить в репорт (минимум для actionable triage)

Без этих 5 пунктов triage задержится, а в edge-case мы закроем репорт как **incomplete**:

1. **Affected version** — точный SHA / tag (например `d9d0d6a`, `v0.2.0-rc1`). Без этого мы не можем воспроизвести.
2. **Reproduction step-by-step** — минимальный Python snippet **или** `workflow_api.json` для ComfyUI, который показывает проблему.
3. **Environment** — Python version, `torch.__version__`, NVIDIA driver (`nvidia-smi`), ОС (Windows / Linux / Colab / Kaggle).
4. **Impact** — что **конкретно** вы можете сделать с этой уязвимостью (RCE / OOB-read / silent fallback / dataleak / DoS). Без impact-описания triage-sexperience = "we don't know severity".
5. **Optional but encouraged** — ссылка на upstream CVE, если уже опубликован; имя/handle для credit в release-notes (если хотите быть отмеченным).

---

## Что ожидать после репорта (Response Timeline)

| Стадия | Deadline | Что делаем |
|:-------|:---------|:-----------|
| **Acknowledgment** | **≤ 7 дней** | Подтверждаем получение, даём initial severity (critical / high / medium / low / not-a-vuln / incomplete) |
| **Investigation** | **≤ 30 дней** | Воспроизводим на нашем CI-staging (T4×2 Kaggle / RTX 4090×2 / A5000+3090), готовим patch в private fork |
| **Coordinated disclosure** | договариваемся | Если репортёр хочет CVE — обсуждаем timeline (стандарт: 90 дней). Если нет — публикуем GHSA + release ASAP. |
| **Public release** | после fix | `v0.2.x` patch-релиз + GitHub Security Advisory (публичная) + обновление `CHANGELOG.md` § `### Security` |

Если репорт признан **"not-a-vulnerability"**, мы ответим с обоснованием (например: "документированное поведение", "upstream-cause", "не наш код"). Функциональный баг при этом **не теряется** — отдельно заводим public Issue с `bug` label.

---

## Out of Scope — что мы НЕ считаем уязвимостью

Чтобы triage был быстрым, вот что **не нужно** репортить сюда:

- **Баги в upstream-зависимостях** (`gguf`, `safetensors`, `comfy-core`, `pollockjj/ComfyUI-MultiGPU`). Репортите upstream, не нам.
- **Баги в model-весах** (LTX 2.3 / Gemma 3). Это license / content-policy, не security — пишите авторам модели.
- **Документированное поведение с WARN**: single-GPU fallback, `effective_donor == primary_dev`, `secondary_dev == primary_dev`. Это **by-design** UX — см. README «Советы и грабли» + `nnodes.py` § `verbose_log`.
- **Performance issues / OOM**. На 2×T4 (16 ГБ × 2) с LTX 2.3 22B GGUF мы **ожидаем** OOM на больших разрешениях — это **не** security, это memory-budget issue.
- **Запросы на фичи / UX-предложения**. Используйте [GitHub Discussions](https://github.com/THE-ANGEL-AI/ComfyUI-LTX2-MultiGPU/discussions) или Issues с `enhancement` label.

---

## Downstream GPL-3.0 copyleft — что это значит для security-репортов в форках

Все downstream-форки, склонированные **после 2026-06-30** (release `v0.2.0`), наследуют **GPL-3.0-or-later** obligations:

* Если вы обнаружили уязвимость в форке **под GPL-3.0** и решаете её — вы **обязаны** либо сообщить upstream (нас), либо опубликовать fix в своём форке под GPL-3.0 в течение разумного срока. «Тихий» fix в проприетарной ветке, скрывающий уязвимость — это конфликт с **GPL §2 / §5** (Conveying Modified Source Versions).
* Если вы форкнули **до 2026-06-30** (под MIT) — лицензия **вашего снимка** остаётся MIT (perpetual grant от старой лицензии), и вы сами решаете, как обрабатывать уязвимости.

Подробности — в `LICENSE` (полный текст GPL-3.0) и § `### Security (copyleft implications)` в `CHANGELOG.md`.

---

## Версия этой политики

* **v1.0** — initial publication (2026-06-30), синхронно с release v0.2.0 / `CHANGELOG.md` §Security.

Maintainer: **The Angel Studio** · Contact: `gi.the.angel@gmail.com` · Repo: `https://github.com/THE-ANGEL-AI/ComfyUI-LTX2-MultiGPU`

`SECURITY.md` распространяется под **GPL-3.0-or-later** (как и код проекта).
