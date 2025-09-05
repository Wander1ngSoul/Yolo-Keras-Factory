import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

class DataLoader:
    @staticmethod
    def load_data_from_folder(dataset_dir, max_samples_per_class=None):
        images, labels = [], []

        for class_name in sorted(os.listdir(dataset_dir)):
            if class_name.isdigit() and int(class_name) in range(10):
                class_dir = os.path.join(dataset_dir, class_name)
                class_label = int(class_name)

                if not os.path.isdir(class_dir):
                    continue

                image_files = [f for f in os.listdir(class_dir)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                if max_samples_per_class:
                    image_files = image_files[:max_samples_per_class]

                for image_name in image_files:
                    image_path = os.path.join(class_dir, image_name)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                    if image is None:
                        print(f"Ошибка загрузки: {image_path}")
                        continue

                    image = cv2.resize(image, (32, 20))
                    images.append(image)
                    labels.append(class_label)

        images = np.array(images, dtype=np.float32) / 255.0
        images = images.reshape(-1, 20, 32, 1)
        labels = np.array(labels)

        print(f"Загружено {len(images)} изображений из {dataset_dir}")
        return images, labels

    @staticmethod
    def prepare_datasets(images, labels, test_size=0.2):
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=test_size, random_state=42, stratify=labels
        )
        y_train = to_categorical(y_train, num_classes=10)
        y_val = to_categorical(y_val, num_classes=10)
        return X_train, X_val, y_train, y_val