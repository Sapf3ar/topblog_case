# topblog_case

## Бэк:
1) Ассинхронная обработка

## Варианты загрузки картинок:
1) Отдельная картинка
2) Ссылка на яндекс диск
3) zip файл с картинками
   
## ML Модель:
1) Классифицирует соцсеть c качеством F1 > 0.99
2) easyocr кластеризует картинку по заданному числу кластеров и вынимает из них текст
3) С помощью графовой кластеризации находит по евклидовому расстоянию ближайшие к ключевому слову соцсети значения метрики
4) Собирает метрики с каждой картинки, а также собирает внутренние значения для поиска ошибок трёх родов
5) Значения выгружатся в xlsx и SQL таблицы

## Фронт

## Визуализация
