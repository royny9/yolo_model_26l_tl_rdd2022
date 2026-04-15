# from ultralytics import YOLO

# model = YOLO('yolov8l.pt') 

# from ultralytics import YOLO


# if __name__ == '__main__':
#     # Загружаем модель
#     model = YOLO('yolov8l.pt') 

#     # Запускаем обучение
#     results = model.train(
#         # --- Базовые настройки ---
#         data="data.yaml", 
#         epochs=100,         
#         imgsz=640,          
#         batch=10,            
        
#         # --- Аугментация ---
#         mosaic=1.0,
#         fliplr=0.5,
#         hsv_h=0.015,
#         hsv_s=0.5,
#         hsv_v=0.4,
#         translate=0.1,
#         scale=0.5,

#         # --- Веса потерь ---
#         box=7.5,            
#         cls=2.5,            
        
#         # --- Железо ---
#         device=0,
#         amp=True,
#         workers=0  
#     )


from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('runs/detect/train32/weights/best.pt')

    # 2. Запускаем "финишную доводку"
    results = model.train(
        # --- Основные настройки ---
        data="data.yaml", 
        epochs=30,           # 30 эпох достаточно для дообучения
        imgsz=800,           # Повышаем детализацию для мелких ям
        batch=4,             # Уменьшено до 4
        
        # --- Тюнинг обучения (бережное привыкание) ---
        optimizer='AdamW',   # Более стабильный для дообучения
        lr0=0.0001,          # Низкий шаг (в 100 раз меньше дефолта)
        lrf=0.01,            # Финальный шаг будет еще меньше
        warmup_epochs=3,     # 3 эпохи "разминки" для адаптации к 800px
        warmup_bias_lr=0.0,
        
        # --- Улучшение качества ---
        box=7.5,             # Сохраняем фокус на точности рамок
        cls=2.5,             # Сохраняем фокус на различении классов
        close_mosaic=5,      # ОТКЛЮЧИТЬ мозаику на последних 5 эпохах для точности
        
        # --- Аугментация (добавляем "сложности") ---
        mixup=0.1,           # Смешивание изображений
        erasing=0.4,         # Имитация частично закрытых объектов (грязь/листья)
        fliplr=0.5,
        
        # --- Железо ---
        device=0,            #
        workers=2,           # Умеренная нагрузка на CPU и RAM
        amp=True             # Смешанная точность для экономии памяти
    )
