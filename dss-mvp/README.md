# DSS MVP

Автономный MVP программного прототипа системы поддержки принятия решений для планирования регламентных работ производственного оборудования на основе открытого набора данных NASA C-MAPSS.

## Назначение

MVP выполняет полный аналитический сценарий без REST API, веб-интерфейса и Docker:

- загружает данные C-MAPSS из папки `datasets`;
- синхронизирует справочники оборудования и телеметрических параметров с Supabase PostgreSQL;
- сохраняет train-траектории в `telemetry.telemetry_measurements`;
- обучает и сохраняет модели прогнозирования остаточного ресурса для `FD001` и `FD002`;
- переиспользует сохранённые модели без повторного обучения;
- рассчитывает прогнозы RUL для тестовых двигателей;
- сохраняет диагностику, прогнозы и рекомендации в схему `app`;
- формирует текстовые отчёты по каждому эксперименту.

## Требования

- Python 3.11+
- доступ к Supabase PostgreSQL
- файлы C-MAPSS должны лежать в папке `datasets`

## Подготовка

1. Создать виртуальное окружение:

```powershell
python -m venv .venv
```

2. Активировать окружение и установить зависимости:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Создать `.env` на основе `.env.example` и заполнить параметры подключения к Supabase PostgreSQL.

## Предустановленные режимы

| Эксперимент | Обучение | Тестирование | Роль в дипломе |
| --- | --- | --- | --- |
| 1 | `train_FD001` | `test_FD001` | Базовая проверка MVP |
| 2 | `train_FD002` | `test_FD002` | Проверка на более сложном сценарии |
| 3 | `train_FD002` | `test_FD001` | Дополнительная проверка переносимости |

## Запуск

По умолчанию запускается `Эксперимент 1`.

```powershell
python src/run_cmapss_mvp.py
```

Явный запуск отдельных режимов:

```powershell
python src/run_cmapss_mvp.py --experiment 1
python src/run_cmapss_mvp.py --experiment 2
python src/run_cmapss_mvp.py --experiment 3
```

Принудительное переобучение и повторная загрузка train-телеметрии:

```powershell
python src/run_cmapss_mvp.py --experiment 2 --retrain --reload-train-telemetry
```

Пользовательский запуск с явным указанием датасетов:

```powershell
python src/run_cmapss_mvp.py --train-dataset-id FD002 --test-dataset-id FD001
```

Запуск на новом test-файле в формате C-MAPSS:

```powershell
python src/run_cmapss_mvp.py --experiment 1 --test-file datasets/test_FD001.txt --rul-file datasets/RUL_FD001.txt
```

## Артефакты

- сохранённые модели:
  - `models/rul_model_fd001.joblib`
  - `models/rul_model_fd002.joblib`
- отчёты:
  - `reports/cmapss_mvp_report.txt` — отчёт последнего запуска
  - `reports/cmapss_mvp_report_experiment_1.txt`
  - `reports/cmapss_mvp_report_experiment_2.txt`
  - `reports/cmapss_mvp_report_experiment_3.txt`

## Примечания

- В рамках MVP один цикл C-MAPSS интерпретируется как один час работы оборудования.
- Train-траектории используются для обучения модели и загрузки телеметрии.
- Test-траектории используются для проверки модели и сохранения прогнозов RUL.
- Для `Эксперимента 3` используется модель, обученная на `FD002`, а прогноз строится для `FD001`.
