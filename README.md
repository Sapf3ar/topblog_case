# miss misis TopBlog case
Дабы облегчить работу сотрудников 'ТопБЛОГ', нами был разработан сервис для автоматической фотоаналитики по распознаванию необходимых показателей участников проекта и корректное занесение их в таблицу и базу данных.

Сервис поможет существенно освободить время работников, улучшить точность, а также эффективнее отслеживать прогресс участников проекта.

Стек: FastApi, Streamit, easyocr, python

Уникальность

Простота, прирост точности, масштабируемость

## Бэк:
1) Ассинхронная обработка
2) Запись результатов в xlsx и SQL таблицы
3)  Выгрузкa xlsx таблиц результатов
4) Выгрузка xlsx таблиц ошибок

## Варианты загрузки картинок:
1) Отдельная картинка
2) Ссылка на яндекс диск
3) zip файл с картинками
   
## ML Модель:
1) Классифицирует соцсеть
2) easyocr выделяет регионы текста и распознаёт его
3) С помощью кластеризации строятся графы и находится значения метрики платформы
4) Собирает метрики с каждой картинки, а также собирает внутренние значения для поиска ошибок трёх родов

   <img width="495" alt="Screenshot 2023-08-27 at 10 13 29" src="https://github.com/Sapf3ar/topblog_case/assets/70803676/e6e1297c-2ce9-4d7a-88bb-83ae225ab7eb">

## Фронт:
 

## Визуализация:
1) Сбор метрик для отдельного пользователя
2) Ошибок пользователя
3) Интерпретация алгоритма графовой кластеризации

   
## Архитектура:
<img width="550" alt="Screenshot 2023-08-27 at 10 14 12" src="https://github.com/Sapf3ar/topblog_case/assets/70803676/fc88b733-b1e1-4225-b579-36170a0ad88f">

## Страктура и запуск:
.
├── back
│   ├── Dockerfile
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   ├── requirements.txt
│   ├── trilinear.py
│   └── yadisk.py
│   
├── Dockerfile
├── features.md
├── front
│   ├── Dockerfile
│   ├── front.py
│   ├── pages
│   │   └── statistics.py
│   └── requirements.txt
├── __init__.py
├── pyrightconfig.json
├── README.md
├── requirements.txt
├── temp
│   ├── broken.csv
│   └── data.xlsx
├── temp_data.xlsx
└── tutorial.db

Запуск бека   : - uvicorn back.main:app
Запуск фронта : - streamlit run front/front.py
