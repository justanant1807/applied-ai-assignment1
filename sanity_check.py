import cv2, numpy as np
from PIL import Image
from pathlib import Path

p = Path(r'.\bin-images\00001.jpg')
print('Exists:', p.exists(), 'Size:', p.stat().st_size if p.exists() else -1)

im1 = cv2.imread(str(p), 1)
print('cv2.imread:', type(im1), getattr(im1, 'shape', None))

try:
    data = np.fromfile(str(p), dtype=np.uint8)
    im2 = cv2.imdecode(data, 1) if data.size>0 else None
    print('cv2.imdecode:', type(im2), getattr(im2, 'shape', None))
except Exception as e:
    print('cv2.imdecode EXC:', e)

try:
    with Image.open(p) as pil:
        pil = pil.convert('RGB')
        arr = np.array(pil)
        print('PIL.open:', type(arr), arr.shape, arr.dtype)
except Exception as e:
    print('PIL.open EXC:', e)
