# Rule-Based Classifier

Детерминированный классификатор уровня опасности configuration drift в Kubernetes на явных правилах.
Каждому drift-событию (diff между `desired` и `live` манифестом) присваивается класс опасности от 1 до 4.
Класс события — максимум по классам операций diff.

| Класс | Смысл |
|---|---|
| 1 | Низкий: поведение, доступность и безопасность не меняются |
| 2 | Средний: меняется поведение или конфигурация, прямой угрозы нет |
| 3 | Высокий: возможна недоступность сервиса или ослаблена защита |
| 4 | Критический: прямой путь к компрометации узла, кластера или учётных данных |

## Как это работает

- Правила описаны декларативно в [`rules/rules.yaml`](rules/rules.yaml): идентификатор, описание, категория, виды ресурсов, шаблон пути (JSON Pointer) и условия на `old`/`new`.
- Уровень опасности определяется только категорией правила и окружением (`dev`/`prod`):

  | Категория | dev | prod |
  |---|---|---|
  | `security_direct` | 4 | 4 |
  | `security_weakening` | 3 | 3 |
  | `availability` | 2 | 3 |
  | `behavior` | 2 | 2 |
  | `cosmetic` | 1 | 1 |

- Для каждой операции применяется первое совпавшее правило (категории проверяются по убыванию опасности). Операция без правила получает класс 2 (`FALLBACK`).
- Значения ресурсов (`100m`, `128Mi`, `2Gi`) сравниваются через парсер Kubernetes quantity, а не как строки.

## Структура

```
classifier/     движок и парсер quantity
rules/          файл правил
prepare_data.py подготовка входных файлов data/input и разметки по сплитам
predict.py      предсказания: python predict.py --split dev|test|all
evaluate.py     метрики: python evaluate.py --split dev|test
tests/          pytest-тесты
data/           входные данные и эталонные метки
results/        предсказания, метрики, замороженные правила и их SHA-256
```

## Запуск

```bash
pip install -r requirements.txt
python predict.py --split test
python evaluate.py --split test
python -m pytest tests
```

Корпус DriftBench-Offline (датасет и размеченные данные) опубликован в репозитории [DriftBench-Offline-Pub](https://github.com/VityazevEgor/DriftBench-Offline-Pub/tree/main). Для сборки входных данных и подсчёта срезов используются его файлы:

- [`dataset/raw`](https://github.com/VityazevEgor/DriftBench-Offline-Pub/tree/main/dataset/raw) — кейсы (`desired.yaml`, `diff.json`, `meta.json`);
- [`dataset/splits/cases.json`](https://github.com/VityazevEgor/DriftBench-Offline-Pub/blob/main/dataset/splits/cases.json) — разбиение на dev и test;
- [`dataset/index.csv`](https://github.com/VityazevEgor/DriftBench-Offline-Pub/blob/main/dataset/index.csv) — метаданные кейсов для срезов в `evaluate.py`;
- [`annotated_dataset/annotation_main_labeled.csv`](https://github.com/VityazevEgor/DriftBench-Offline-Pub/blob/main/annotated_dataset/annotation_main_labeled.csv) — эталонная разметка;
- [`annotated_dataset/annotation_second_labeled.csv`](https://github.com/VityazevEgor/DriftBench-Offline-Pub/blob/main/annotated_dataset/annotation_second_labeled.csv) — разметка второго эксперта.

Готовые результаты подготовки лежат в `data/`, поэтому для `predict.py` корпус не нужен. `prepare_data.py` и `evaluate.py` читают корпус из каталога `DriftBench-Offline-Pub`, расположенного рядом с этим репозиторием, либо из пути в переменной окружения `DRIFTBENCH_DIR`:

```bash
git clone https://github.com/VityazevEgor/DriftBench-Offline-Pub.git ../DriftBench-Offline-Pub
python prepare_data.py
python evaluate.py --split test
```

## Результаты на test (252 кейса)

| Метрика | Значение |
|---|---|
| accuracy | 0.940 |
| macro-F1 | 0.941 |
| κ Коэна (линейно взвешенная) | 0.934 |

Подробные метрики: `results/metrics_test.json`. SHA-256 замороженных правил: `results/rules_frozen.sha256`.
