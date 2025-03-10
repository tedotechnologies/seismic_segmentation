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
Скачать [segment-anything-2](https://github.com/huggingface/segment-anything-2?tab=readme-ov-file)

## Описание данных в папке: /data  
Salt2d - 2d изображения (реальные) соли — dice 0.66  
sabamrine - 50 синтетических 3d кубов палеорусла (256, 256, 256) отсюда: [Zenodo](https://zenodo.org/records/11079950) — dice 0.15  
paleokart - 120 синтетических 3d снимков палеокарт: [KarstSeg3D](https://github.com/xinwucwp/KarstSeg3D?ysclid=m6thujhkyd532854217)  
!!!! sam2.1_hiera_base_plus.pt модель не справляется совсем — dice 0.21

## Дополнительная информация
- На текущий момент все эксперименты проводятся на модели **sam** (facebook/sam-vit-huge).
- Запуск обучения происходит путем вызова команды:

./run_train.sh
- Предварительно необходимо проверить конфигурационные файлы в папке **conf**.
- Убедитесь, что данные скачаны и имеется доступ к **ClearML**.