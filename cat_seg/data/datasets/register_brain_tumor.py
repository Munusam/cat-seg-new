import os

from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import load_sem_seg

BRAIN_TUMOR_CLASSES = [
    "background",
    "brain tumor"
]


def _get_meta():
    return {
        "stuff_classes": BRAIN_TUMOR_CLASSES,
    }


def register_all_brain_tumor(root):

    root = os.path.join(root, "brain_tumor")

    meta = _get_meta()

    splits = [
        (
            "brain_tumor_train_sem_seg",
            "images/train",
            "annotations_detectron2/train",
        ),
        (
            "brain_tumor_val_sem_seg",
            "images/val",
            "annotations_detectron2/val",
        ),
    ]

    for name, image_dirname, sem_seg_dirname in splits:

        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)

        DatasetCatalog.register(
            name,
            lambda x=image_dir, y=gt_dir:
            load_sem_seg(
                y,
                x,
                gt_ext="png",
                image_ext="png"
            ),
        )

        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )


_root = os.getenv(
    "DETECTRON2_DATASETS",
    "datasets"
)

register_all_brain_tumor(_root)