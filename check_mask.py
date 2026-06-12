from PIL import Image
import numpy as np

mask = np.array(
    Image.open(
        r"E:\new_catseg\CAT-Seg\datasets\brain_tumor\annotations_detectron2\train\TCGA_CS_4941_19960909_13.png"
    )
)

print(mask.shape)
print(np.unique(mask))