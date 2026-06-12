import os
import random
import shutil
import numpy as np

from PIL import Image
from sklearn.model_selection import train_test_split

SOURCE_DIR = r"C:\Users\samy3\Downloads\archive (3)\kaggle_3m"

TARGET_DIR = r"E:\new_catseg\CAT-Seg\datasets\brain_tumor"

IMAGE_TRAIN = os.path.join(TARGET_DIR, "images", "train")
IMAGE_VAL = os.path.join(TARGET_DIR, "images", "val")

MASK_TRAIN = os.path.join(
    TARGET_DIR,
    "annotations_detectron2",
    "train"
)

MASK_VAL = os.path.join(
    TARGET_DIR,
    "annotations_detectron2",
    "val"
)

for d in [
    IMAGE_TRAIN,
    IMAGE_VAL,
    MASK_TRAIN,
    MASK_VAL
]:
    os.makedirs(d, exist_ok=True)

samples = []

for patient_folder in os.listdir(SOURCE_DIR):

    patient_path = os.path.join(
        SOURCE_DIR,
        patient_folder
    )

    if not os.path.isdir(patient_path):
        continue

    for file in os.listdir(patient_path):

        if (
            file.endswith(".tif")
            and "_mask" not in file
        ):

            image_path = os.path.join(
                patient_path,
                file
            )

            mask_path = os.path.join(
                patient_path,
                file.replace(
                    ".tif",
                    "_mask.tif"
                )
            )

            if os.path.exists(mask_path):

                samples.append(
                    (
                        image_path,
                        mask_path
                    )
                )

print("Total samples:", len(samples))

train_samples, val_samples = train_test_split(
    samples,
    test_size=0.2,
    random_state=42
)

print("Train:", len(train_samples))
print("Val:", len(val_samples))


def process_dataset(
    sample_list,
    image_dir,
    mask_dir
):

    for image_path, mask_path in sample_list:

        name = os.path.basename(
            image_path
        ).replace(
            ".tif",
            ".png"
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image.save(
            os.path.join(
                image_dir,
                name
            )
        )

        mask = np.array(
            Image.open(mask_path)
        )

        mask = (
            mask > 0
        ).astype(np.uint8)

        Image.fromarray(mask).save(
            os.path.join(
                mask_dir,
                name
            )
        )


process_dataset(
    train_samples,
    IMAGE_TRAIN,
    MASK_TRAIN
)

process_dataset(
    val_samples,
    IMAGE_VAL,
    MASK_VAL
)

print("Done!")