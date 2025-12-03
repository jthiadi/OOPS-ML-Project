from ultralytics import YOLO


def main():
    # Load pretrained YOLO11-L detector (COCO weights)
    model = YOLO("yolo11l.pt")

    # Fine-tune on your custom 11-class OOPS dataset:
    # charging-cable, earphones, glasses, ipad, keys, multiple, 
    # paper, pen, student-id, wallet, watch
    model.train(
        data="data_oops.yaml",  # this file defines paths and class names
        epochs=50,              # adjust as needed
        imgsz=960,
        batch=4,
        lr0=0.001,              # initial learning rate
        workers=4,
        project="runs_oops",
        name="yolo11l_oops",
        exist_ok=True,
    )

    # Optional: evaluate on the same data config
    model.val(data="data_oops.yaml")


if __name__ == "__main__":
    main()

