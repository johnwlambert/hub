---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
title: MSeg
summary: The MSeg model for computing semantic segmentation from a single image.
image: intel-logo.jpg
author: Intel ISL
tags: [vision]
github-link: https://github.com/mseg-dataset/mseg-semantic
github-id: mseg-dataset/mseg-semantic
featured_image_1: mseg_samples.png
featured_image_2: no-image
accelerator: cuda
---

```python
import torch
mseg = torch.hub.load("intel-isl/MSeg", "MSeg")
mseg.eval()
```

will load the MSeg 1080p model. The model expects 3-channel RGB images of shape ```(3 x H x W)```. Images are expected to be normalized using
`mean=[0.485, 0.456, 0.406]` and `std=[0.229, 0.224, 0.225]`. 
`H` and `W` need to be divisible by `32`. For optimal results `H` and `W` should be close to `384` (the training resolution). 
We provide a custom transformation that performs resizing while maintaining aspect ratio. 

### Model Description

[MSeg](http://vladlen.info/papers/MSeg.pdf) computes a semantic segmentation label map from a single image. The model has been trained on 7 large, diverse datasets using to ensure high quality on a wide range of domains.


### Example Usage

Download an image from the PyTorch homepage
```python
import cv2
import torch
import urllib.request

import matplotlib.pyplot as plt

url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)
```

Load the model

```python
midas = torch.hub.load("intel-isl/MSeg", "MSeg")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
```


Load transforms to resize and normalize the image
```python
mseg_transforms = torch.hub.load("intel-isl/MSeg", "transforms")
transform = mseg_transforms.default_transform
```

Load image and apply transforms
```python
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_batch = transform(img).to(device)
```


Predict and resize to original resolution
```python
with torch.no_grad():
    prediction = mseg(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    
output = prediction.cpu().numpy()
```

Show result
```python 
plt.imshow(output)
# plt.show()
```

### Reference
[MSeg: A Composite Dataset for Multi-domain Semantic Segmentation](https://arxiv.org/abs/1907.01341)

Please cite our paper if you use our model:
```bibtex
@InProceedings{MSeg_2020_CVPR,
author = {Lambert, John and Zhuang, Liu and Sener, Ozan and Hays, James and Koltun, Vladlen},
title = {{MSeg}: A Composite Dataset for Multi-domain Semantic Segmentation},
booktitle = {Computer Vision and Pattern Recognition (CVPR)},
year = {2020}
}
```
