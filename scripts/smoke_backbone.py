import numpy as np
from PIL import Image

from vla_arch.backbones import Qwen35_2B

bb = Qwen35_2B()
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

feats, mask = bb(images=[[img], [img, img]], texts=["pick up the red cube", "open the drawer"])
print("features:", tuple(feats.shape), "mask:", tuple(mask.shape))
print("chat:", bb.chat(images=[[img]], texts=["Describe this image in one sentence."]))
