import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model, Model

class class_cam:
    """
    Class for performing Grad-CAM (Gradient-weighted Class Activation Mapping)
    on classification models.S
    """

    def __init__(self, model, layer_name):
        """
        Initializes the GradCAM object.

        Parameters:
        model (tensorflow.keras.Model): the trained classification model.
        layer_name (str): the name of the penultimate dense layer of the model.
        """
        self.model = model
        self.layer_name = layer_name

        # Create a new model that outputs the feature maps of the penultimate layer
        self.feature_extraction_model = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(self.layer_name).output
        )

    def generate_heatmap(self, image_array):
        """
        Generates the Grad-CAM heatmap for the input image.

        Parameters:
        image_array (numpy.ndarray): the input image as a numpy array.

        Returns:
        numpy.ndarray: the Grad-CAM heatmap as a numpy array.
        """
        # Expand the image array to have a batch dimension of 1
        image_array = np.expand_dims(image_array, axis=0)

        # Get the predicted class and the feature maps of the penultimate layer
        with tf.GradientTape() as tape:
            features = self.feature_extraction_model(image_array)
            preds = self.model(image_array)

            # Get the index of the predicted class
            pred_index = tf.argmax(preds, axis=1)[0]

            # Get the predicted score for the predicted class
            pred_score = preds[:, pred_index][0]

        # Compute the gradients of the predicted score with respect to the feature maps
        grads = tape.gradient(pred_score, features)

        # Get the channel-wise weights of the feature maps
        channel_weights = tf.reduce_mean(grads, axis=(1, 2))
        channel_weights = tf.expand_dims(channel_weights, axis=-1)

        # Compute the weighted sum of the feature maps and apply ReLU
        heatmap = tf.reduce_sum(features * channel_weights, axis=-1)
        heatmap = tf.nn.relu(heatmap)

        # Normalize the heatmap between 0 and 1
        max_val = tf.math.reduce_max(heatmap)
        if max_val != 0:
            heatmap /= max_val

        # Convert the heatmap to a numpy array
        heatmap = heatmap.numpy()[0]

        return heatmap
