# graddy: A Gradient-weighted Class Activation Mapping Package

Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique used to visualize the importance of different regions of an input image for a specific classification decision made by a convolutional neural network (CNN).

When it comes to using Grad-CAM for segmentation, the main difference is that instead of generating a heatmap for a single class, we generate a heatmap for each pixel of the image, indicating its importance for segmentation. This requires modifying the implementation of Grad-CAM to calculate the gradients of the output segmentation mask with respect to the input image, rather than just the gradients of the output classification score with respect to the convolutional feature maps.

The modified implementation typically involves computing the gradient of the output segmentation mask with respect to the final convolutional layer, then using these gradients to weight the activation maps of this layer to generate a heatmap. The heatmap can then be overlaid on the input image to highlight the regions of the image that are most important for the segmentation.

## Who's your graddy? 🌚

This package provides a simple implementation of Grad-CAM (Gradient-weighted Class Activation Mapping) for visualizing and interpreting the predictions of convolutional neural networks. Grad-CAM is a technique for generating heatmaps that indicate which regions of an input image are most important for the model's prediction. This can be helpful for understanding what features the model is using to make its decision.

graddy specialises in class activation map generation for classification and segmentation, as well as capturing them and your mom.

Refer test_gradcam.py to use the package.

## Installation

You can install this package via pip:

```
pip install graddy
```

#### gradcam Class
The gradcam class in the package can be used to generate Grad-CAM heatmaps for a trained classification model. Here's an example:

```
import numpy as np
import matplotlib.pyplot as plt
from graddy import seg_gradcam
```

## Load your model

```
model_path = "model_path.h5"
gradcam = seg_gradcam(model_path)
```
## Load an example image

```
image_path = "example_image_path.jpeg"
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

- **model_path**: the path to the saved Keras model file.
- **target_size (optional)**: the size to which images should be resized (height, width.Default is (512, 512).
- **preprocess_input (optional)**: a function that will be applied to the image before passing it to the model. Default is None.
- **colormap (optional)**: the colormap to use for the heatmap. Default is "jet".
- **alpha (optional)**: the transparency of the heatmap. Should be between 0 and 1. Default is 0.5.
