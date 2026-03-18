from ultralytics import YOLO

def main():
    model = YOLO("yolo26l.pt")


    results =   model.train(
        data="data.yaml", 
        epochs=30, 
        imgsz=388, 
        batch=16, 
        workers=4, 
        device=0,  # Явно указываем GPU
        exist_ok=True
    )


    results = model("E:/programming/y/RDD_SPLIT/test/images/China_Drone_000550.jpg")

if __name__ == '__main__':
    main()