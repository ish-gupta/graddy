import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model, Model

class seg_cam:
    """
    A class to generate Grad-CAM heatmaps and superimpose them on images.

    Parameters
    ----------
    model_path : str
        The path to the saved Keras model file.
    target_size : tuple, optional
        The size to which images should be resized (height, width).
        Default is (512, 512).
    preprocess_input : function, optional
        A function that will be applied to the image before passing it to the model.
        Default is None.
    colormap : str, optional
        The colormap to use for the heatmap. Default is "jet".
    alpha : float, optional
        The transparency of the heatmap. Should be between 0 and 1. Default is 0.5.
    """
    def __init__(self, model_path, target_size=(512, 512), preprocess_input=None, colormap="jet", alpha=0.5):
        self.model_path = model_path
        self.target_size = target_size
        self.preprocess_input = preprocess_input
        self.colormap = colormap
        self.alpha = alpha
        
        self.model = load_model(self.model_path)
        self.grad_model = None
    
    def read_image(self, path):
        """
        Read an image from a file and preprocess it.

        Parameters
        ----------
        path : str
            The path to the image file.

        Returns
        -------
        numpy.ndarray
            The preprocessed image as a NumPy array.
        """
        img = load_img(path, color_mode="rgb", target_size=self.target_size)
        img = img_to_array(img)
        if self.preprocess_input:
            img = self.preprocess_input(img)
        else:
            img = img / 255.
        return img
    
    def generate_heatmap(self, img_array, layer_index):
        """
        Generate a Grad-CAM heatmap for a given image and layer.

        Parameters
        ----------
        img_array : numpy.ndarray
            The preprocessed image as a NumPy array.
        layer_index : int
            The index of the layer to use for generating the heatmap.

        Returns
        -------
        numpy.ndarray
            The heatmap as a NumPy array.
        """
        if self.grad_model is None:
            self.grad_model = Model(inputs=self.model.inputs,
                                    outputs=[self.model.layers[layer_index].output, self.model.output])

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = self.grad_model(img_array)
            class_channel = tf.reduce_sum(preds[preds >= 0.5])[..., tf.newaxis]

        grads = tape.gradient(class_channel, last_conv_layer_output)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()

        return heatmap
    
    def generate_heatmap_image(self, image_path, layer_name=None, class_index=None, alpha=0.4, save_path=None):
        """
        Generates a heatmap overlay image of the input image with the Grad-CAM
        heatmap.

        Args:
            image_path (str): Path to the input image file.
            layer_name (str, optional): Name of the convolutional layer to
                compute the Grad-CAM heatmap for. If not provided, the last
                convolutional layer is used. Defaults to None.
            class_index (int, optional): Index of the class to compute the
                Grad-CAM heatmap for. If not provided, the predicted class is
                used. Defaults to None.
            alpha (float, optional): Alpha value for blending the heatmap with
                the input image. Defaults to 0.4.
            save_path (str, optional): Path to save the heatmap overlay image
                to. If not provided, the image is not saved. Defaults to None.

        Returns:
            numpy.ndarray: A numpy array of the heatmap overlay image in RGB
                format.
        """

        # Load the input image
        img = load_img(image_path, target_size=self.target_size)
        img = img_to_array(img) / 255.

        # Compute the Grad-CAM heatmap
        heatmap = self.compute_heatmap(img, layer_name, class_index)

        # Rescale the heatmap to the size of the input image
        heatmap = np.uint8(255 * heatmap)
        heatmap = cm.jet(heatmap)[..., :3]
        heatmap = tf.keras.preprocessing.image.array_to_img(heatmap)
        heatmap = heatmap.resize((img.shape[1], img.shape[0]))
        heatmap = tf.keras.preprocessing.image.img_to_array(heatmap)

        # Blend the heatmap with the input image
        superimposed_img = heatmap * alpha + img
        superimposed_img = np.clip(superimposed_img, 0, 1)

        # Convert the superimposed image to RGB format
        superimposed_img *= 255.
        superimposed_img = np.uint8(superimposed_img)
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
        superimposed_img = np.array(superimposed_img)

        # Save the heatmap overlay image if a save path is provided
        if save_path is not None:
            tf.keras.preprocessing.image.save_img(save_path, superimposed_img)

        return superimposed_img