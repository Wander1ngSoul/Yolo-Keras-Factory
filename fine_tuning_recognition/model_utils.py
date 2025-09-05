from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2

class ModelUtils:
    @staticmethod
    def create_conservative_model(base_model_path):
        print(f"Загрузка базовой модели: {base_model_path}")
        base_model = load_model(base_model_path)

        for layer in base_model.layers[:-4]:
            layer.trainable = False

        for layer in base_model.layers[-4:]:
            layer.trainable = True
            print(f"Разморожен слой: {layer.name}")

        optimizer = Adam(learning_rate=0.0001)

        base_model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print("Модель готова для консервативного дообучения")
        return base_model

    @staticmethod
    def robust_augmentation(image):
        if len(image.shape) == 3 and image.shape[2] == 1:
            image = image[:, :, 0]

        if np.random.random() > 0.3:
            noise = np.random.normal(0, 0.05, image.shape)
            image = image + noise
            image = np.clip(image, 0, 1)

        if np.random.random() > 0.5:
            kernel_size = np.random.choice([3, 5, 7])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        if np.random.random() > 0.3:
            brightness = np.random.uniform(0.7, 1.4)
            image = image * brightness
            image = np.clip(image, 0, 1)

            contrast = np.random.uniform(0.7, 1.4)
            image = 0.5 + contrast * (image - 0.5)
            image = np.clip(image, 0, 1)

        if np.random.random() > 0.4:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) / 9.0
            image = cv2.filter2D(image, -1, kernel)
            image = np.clip(image, 0, 1)

        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]

        return image