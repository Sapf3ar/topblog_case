Для повышения эффективности команды "ТопБЛОГ" мы создали сервис автоматической фотоаналитики. Этот инструмент распознает ключевые показатели участников проекта, возвращает результат в виде таблицы и заносит информацию в базу данных.

С его помощью сотрудники "ТопБЛОГ" могут сосредоточиться на более важных задачах, гарантируя точность данных и оперативное отслеживание успехов участников.

Технологический стек: Python3, FastApi, Streamlit, EasyOCR, Scikit-learn.


Уникальность нашего решения заключается в трех ключевых аспектах:

- Простота: Интуитивно понятный интерфейс и легкость интеграции обеспечивают комфортное взаимодействие с сервисом.
- Точность: Наш алгоритм, базируясь на передовых технологиях, гарантирует высокую степень точности в распознавании данных.
- Масштабируемость: Платформа разработана с учетом потребностей роста и развития, обеспечивая стабильную работу при увеличении объемов данных и пользователей.

## Функционал бэкенда:
1) Ассинхронная обработка данных
2) Запись результатов в xlsx и SQL таблицы
3) Экспорт результатов и диагностика ошибок в формате xlsx таблиц

## Методы загрузки картинок:
1) Загрузка единичного изображения
2) Вставка ссылки на Яндекс.Диск
3) Загрузка zip-архива изображений
   
## Модель машинного обучения:
1) Классификация социальной сети и сервиса
2) Использование EasyOCR для детекции и распознавания текста
3) Применение методов кластеризации для создания графов и определения значений метрики для платформы
4) Агрегация метрик с каждого изображения и сбор внутренних показателей для выявления трех родов ошибок.

   <img width="495" alt="Screenshot 2023-08-27 at 10 13 29" src="https://github.com/Sapf3ar/topblog_case/assets/70803676/e6e1297c-2ce9-4d7a-88bb-83ae225ab7eb">

## Фронтенд:
 
## Визуализация:
1) Сбор метрик для отдельного пользователя
2) Ошибок пользователя
3) Интерпретация алгоритма графовой кластеризации

   
## Архитектура:
<img width="550" alt="Screenshot 2023-08-27 at 10 14 12" src="https://github.com/Sapf3ar/topblog_case/assets/70803676/fc88b733-b1e1-4225-b579-36170a0ad88f">

## Запуск бэкенда
uvicorn back.main:app

## Запуск фронтенда
streamlit run front/front.py
