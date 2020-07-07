"""A sample module of loading the TFLite model of FaceMesh from Google."""
import cv2
import tensorflow as tf


class MeshDetector(object):
    """Face mesh detector"""

    def __init__(self, model_path):
        """Initialization"""
        # Initialize the input image holder.
        self.target_image = None

        # Load the model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)

        # Set model input
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def _preprocess(self, image_bgr):
        """Preprocess the image to meet the model's input requirement.
        Args:
            image_bgr: An image in default BGR format.

        Returns:
            image_norm: The normalized image ready to be feeded.
        """
        image_resized = cv2.resize(image_bgr, (192, 192))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_norm = (image_rgb-127.5)/127.5
        return image_norm

    def get_mesh(self, image):
        """Detect the face mesh from the image given.
        Args:
            image: An image in default BGR format.

        Returns:
            mesh: A face mesh, normalized.
            score: Confidence score.
        """
        # Preprocess the image before sending to the network.
        image = self._preprocess(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = image[tf.newaxis, :]

        # The actual detection.
        self.interpreter.set_tensor(self.input_details[0]["index"], image)
        self.interpreter.invoke()

        # Save the results.
        mesh = self.interpreter.tensor(self.output_details[0]["index"])()[
            0].reshape(468, 3) / 192
        score = self.interpreter.tensor(self.output_details[1]["index"])()[0]

        return mesh, score

    def draw_mesh(self, image, mesh):
        """Draw the mesh on an image"""
        # The mesh are normalized which means we need to convert it back to fit
        # the image size.
        image_size = image.shape[0]
        mesh *= image_size
        for point in mesh:
            cv2.circle(image, (point[0], point[1]), 2, (0, 255, 128), -1)

        # TODO: draw the conter lines.
