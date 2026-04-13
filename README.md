# Цифровой двойник 

## Описание
Проект моделирует Shannon Index с использованием Random Forest.

## Требования
Python 3.10+

## Установка

git clone https://github.com/GnomGenome/digital-twin-soil/tree/main
cd Цифровой_двойник

python -m venv venv

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt

## Структура

- Data/ — входные данные
- Results/ — результаты
- Scripts/ — скрипты

## Порядок запуска

ВАЖНО: запускать по порядку:

1. Скрипт 01
2. Скрипт 03
3. Scripts/04_model_random_forest.py

## Запуск

python Scripts/04_model_random_forest.py

## Результаты

Сохраняются в папку Results:
- feature_importance_full.csv
- importance_microbes.csv
- importance_soil.csv
- график важности признаков
