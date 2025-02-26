# rosneft_segmentation

Проект по сегментации изображений для Роснефть

## Используемый стек
uv + clearml

## Пайплайн
- подготовка данных
- обучение
- валидация

## Управление
Управлять всем через hydra + clearml


## Перед началом работы
скачать https://github.com/huggingface/segment-anything-2?tab=readme-ov-file


## Описание данных в папке: /data
Salt2d - 2d изображения (реальные) соли
sabamrine - пока 50 3d синтетически кубов палеорусла отсюда: https://zenodo.org/records/11079950
paleokart - 120 синтетических 3d снимков палеокарт: https://github.com/xinwucwp/KarstSeg3D?ysclid=m6thujhkyd532854217



