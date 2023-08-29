# Сервис автоматической фотоаналитики "ТопБЛОГ"

Для повышения эффективности команды "ТопБЛОГ" мы создали сервис автоматической фотоаналитики. Этот инструмент распознает ключевые показатели участников проекта, возвращает результат в виде таблицы и заносит информацию в базу данных.

С его помощью сотрудники "ТопБЛОГ" могут сосредоточиться на более важных задачах, гарантируя точность данных и оперативное отслеживание успехов участников. 

Технологический стек: Python3, FastApi, Streamlit, EasyOCR, Scikit-learn.


Уникальность нашего решения заключается в трех ключевых аспектах:

- Простота: Интуитивно понятный интерфейс и легкость интеграции обеспечивают комфортное взаимодействие с сервисом.
- Точность: Наш алгоритм, базируясь на передовых технологиях, гарантирует высокую степень точности в распознавании данных.
- Масштабируемость: Платформа разработана с учетом потребностей роста и развития, обеспечивая стабильную работу при увеличении объемов данных и пользователей.

![image](https://github.com/Sapf3ar/topblog_case/assets/108126763/c4ad713b-e88b-4d46-b898-c6946460e370)

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

![image](https://github.com/Sapf3ar/topblog_case/assets/108126763/fac3223b-c272-41a3-821d-6342785d275c)


## Фронтенд:
 
## Визуализация:
1) Сбор метрик для отдельного пользователя
2) Ошибок пользователя
3) Интерпретация алгоритма графовой кластеризации 

   
## Архитектура:

![image](https://github.com/Sapf3ar/topblog_case/assets/108126763/38dba222-e6f3-42cf-a868-cf643da12178)

## Запуск бэкенда
```
uvicorn back.main:app
```

## Запуск фронтенда
```
streamlit run front/front.py
```
