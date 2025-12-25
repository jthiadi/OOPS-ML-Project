from ultralytics import YOLO
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))


def main():
    model = YOLO("yolo11l.pt")
    model.train(
        data="data_oops.yaml",  
        epochs=50,             
        imgsz=960,
        batch=4,
        lr0=0.001,              
        workers=4,
        project="runs_oops",
        name="yolo11l_oops",
        exist_ok=True,
        freeze=15,
        save_period=5,
    )

    model.val(data="data_oops.yaml")


if __name__ == "__main__":
    main()

