from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from Grad_CAM.graddy import seg_gradcam

# Instantiate the Seg_GradCam class
model_path = "tests/unet_agriculture_v3.h5"
gradcam = seg_gradcam(model_path)

# Load an example image
image_path = "tests/236_good.jpeg"
image = Image.open(image_path)

# Generate the heatmap overlay image
heatmap_image = gradcam.generate_heatmap_image(image_path)

# Display the heatmap overlay image
plt.imshow(heatmap_image)
plt.axis("off")
plt.show()