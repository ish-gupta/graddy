# Grad_CAM: A Gradient-weighted Class Activation Mapping package

This package provides a simple implementation of Grad-CAM (Gradient-weighted Class Activation Mapping) for visualizing and interpreting the predictions of convolutional neural networks. Grad-CAM is a technique for generating heatmaps that indicate which regions of an input image are most important for the model's prediction. This can be helpful for understanding what features the model is using to make its decision.

Refer test_gradcam.py to use the package.

## Installation

You can install this package via pip:

```
pip install Grad_CAM
```

#### gradcam Class
The gradcam class in the package can be used to generate Grad-CAM heatmaps for a trained classification model. Here's an example:

```
import numpy as np
import matplotlib.pyplot as plt
from Grad_CAM.graddy import seg_gradcam
```

## Load your model

```
model_path = "tests/model.h5"
gradcam = seg_gradcam(model_path)
```
## Load an example image

```
image_path = "tests/236_good.jpeg"
image = Image.open(image_path)
```

## Generate the heatmap overlay image


```
heatmap_image = gradcam.generate_heatmap_image(image_path)
```

## Display the heatmap overlay image

```
plt.imshow(heatmap_image)
plt.axis("off")
plt.show()
```
## Arguments for seg_gradcam class
The seg_gradcam class takes the following arguments:

**model_path**: the path to the saved Keras model file.
**target_size (optional)**: the size to which images should be resized (height, width.Default is (512, 512).
**preprocess_input (optional)**: a function that will be applied to the image before passing it to the model. Default is None.
**colormap (optional)**: the colormap to use for the heatmap. Default is "jet".
**alpha (optional)**: the transparency of the heatmap. Should be between 0 and 1. Default is 0.5.


## Requirements
tensorflow
numpy
matplotlib