from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog

import cat_seg.data.datasets

print("\nRegistered datasets:\n")

for name in DatasetCatalog.list():
    if "brain" in name:
        print(name)

print("\nMetadata:\n")

meta = MetadataCatalog.get(
    "brain_tumor_train_sem_seg"
)

print(meta.stuff_classes)

dataset = DatasetCatalog.get(
    "brain_tumor_train_sem_seg"
)

print("\nSamples:", len(dataset))

print("\nFirst sample:\n")
print(dataset[0])