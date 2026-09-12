"""
Step 7: Fine-tune YOLO on an obstacle-detection dataset.

How to get the dataset:
1. Go to https://universe.roboflow.com and search "obstacle detection".
2. Pick a dataset whose classes match what you care about
   (pole, stairs, curb, pothole, person, vehicle, etc.).
3. On the dataset page, use the "Download Dataset" option and choose
   format: "YOLOv8". This gives you a zip containing:
       train/images, train/labels
       valid/images, valid/labels
       data.yaml   <- this file describes the classes and paths
4. Unzip it somewhere, e.g. next to this script as ./obstacle_dataset/
5. Update DATASET_YAML_PATH below to point at that data.yaml.

What this script does:
- Starts from the same yolov8n.pt COCO-pretrained weights you've been
  using (transfer learning) rather than training from scratch - this
  means it keeps everything it already knows (person, car, chair, etc.)
  while learning the new hazard classes from your dataset on top.
- Trains for a set number of epochs and saves the best checkpoint.
- The resulting best.pt is a drop-in replacement for "yolov8n.pt" in
  step6_3d_position.py - just change yolo_model_name to point at it.

Run it with:
    python step7_finetune.py

Training takes a while depending on dataset size and GPU - watch the
console for progress. An RTX 4050 should handle a modest dataset
(a few hundred to a couple thousand images) in well under an hour.
"""

from ultralytics import YOLO

DATASET_YAML_PATH = "./Indoor.yolov8/data.yaml"  # <-- update this path
BASE_MODEL = "yolov8n.pt"  # start from the same checkpoint you've been testing with
EPOCHS = 50
IMAGE_SIZE = 640


def main():
    print(f"[INFO] Loading base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    print(f"[INFO] Starting fine-tuning on: {DATASET_YAML_PATH}")
    print(f"[INFO] This will train for {EPOCHS} epochs at image size {IMAGE_SIZE}.")

    results = model.train(
        data=DATASET_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        device=0,  # GPU index 0 - change to "cpu" if needed
        patience=10,  # stop early if validation performance stalls for 10 epochs
        project="obstacle_finetune",
        name="run1",
    )

    print("[OK] Training complete.")
    print("[INFO] Your fine-tuned weights are saved at:")
    print("       obstacle_finetune/run1/weights/best.pt")
    print("[INFO] Use this path as yolo_model_name in step6_3d_position.py")
    print("       to test your fine-tuned model on the live feed.")


if __name__ == "__main__":
    main()
