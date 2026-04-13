# Цифровой двойник 

## Описание
Проект моделирует Shannon Index с использованием Random Forest.

## Требования
Python 3.13+

## ⚙️ Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/GnomGenome/digital-twin-soil.git
cd digital-twin-soil
```

### 2. Создание виртуального окружения
- Mac / Linux
```bash
python -m venv test_env
source test_env/bin/activate
```
- Windows 
```bash
python -m venv test_env
test_env\Scripts\activate
```
### 3. Установка зависимостей
- Mac/Linux
```bash
pip install -r Scripts/requirements.txt
```
- Windows
```bash
pip install -r Scripts\requirements.txt
```
### 4. Запуск
- Mac/Linux
```bash
python Scripts/04_model_random_forest.py
```
- Windows
```bash
python Scripts\04_model_random_forest.py
```

## Структура

- Data/ — входные данные*
- Results/ — результаты
- Scripts/ — скрипты
* — Входные данные всего пайплайна содержатся в Data, однако входные данные для работы скрипта 04_model_random_forest.py содержатся в папке Results, так как они являются результатом работы других скриптов.
## Результаты

Сохраняются в папку Results:
- feature_importance_full.csv
- importance_microbes.csv
- importance_soil.csv
- график важности признаков
