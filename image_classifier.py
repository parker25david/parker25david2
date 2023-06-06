import tensorflow as tf
from tensorflow import keras
import numpy as np

class ImageClassifier:
    def __init__(self, model_path, class_labels):
        self.model = keras.models.load_model(model_path)
        self.class_labels = class_labels

    def classify_image(self, image_path):
        image = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image = keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = keras.applications.mobilenet.preprocess_input(image)

        predictions = self.model.predict(image)
        predicted_class_index = np.argmax(predictions)
        predicted_class = self.class_labels[predicted_class_index]

        return predicted_class

# Example usage:
model_path = 'path/to/your/model.h5'
class_labels = ['cat', 'dog', 'flower', 'car']

classifier = ImageClassifier(model_path, class_labels)

image_path = 'path/to/your/image.jpg'
predicted_class = classifier.classify_image(image_path)

print("Predicted class:", predicted_class)
