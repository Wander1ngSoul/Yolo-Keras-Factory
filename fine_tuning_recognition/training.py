from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .model_utils import ModelUtils
from .callbacks import get_training_callbacks

class Trainer:
    def __init__(self, results_dir):
        self.results_dir = results_dir

    def train_conservative(self, model, X_train, y_train, X_test, y_test, epochs=20):
        original_loss, original_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Оригинальное качество модели: accuracy={original_accuracy:.4f}")

        callbacks = get_training_callbacks(X_test, y_test, original_accuracy, self.results_dir)

        datagen = ImageDataGenerator(
            rotation_range=8,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.15,
            fill_mode='constant',
            cval=0.0,
            preprocessing_function=ModelUtils.robust_augmentation
        )

        print("Начало консервативного дообучения...")
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=16),
            steps_per_epoch=len(X_train) // 16,
            validation_data=(X_test, y_test),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        return model, history