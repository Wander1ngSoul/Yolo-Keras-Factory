import os
from datetime import datetime
from Recognition.common.config import RecognitionConfig
from Recognition.common.data_loader import DataLoader
from Recognition.common.models import ModelFactory
from Recognition.common.training_utils import TrainingUtils
from Recognition.common.evaluation import Evaluator
from Recognition.common.callbacks import get_training_callbacks
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np


class ConservativeFineTuner:
    def __init__(self, base_model_path, original_dataset_dir, new_dataset_dir):
        self.base_model_path = base_model_path
        self.original_dataset_dir = original_dataset_dir
        self.new_dataset_dir = new_dataset_dir
        self.base_model = None
        self.fine_tuned_model = None
        self.history = None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config = RecognitionConfig()
        self.results_dir = os.path.join(config.FT_RESULTS_DIR, f'conservative_fine_tuning_{timestamp}')
        os.makedirs(self.results_dir, exist_ok=True)

        print(f"Результаты будут сохранены в: {self.results_dir}")

    def prepare_conservative_datasets(self):
        print("Загрузка оригинального датасета...")
        X_orig, y_orig = DataLoader.load_data_from_folder(self.original_dataset_dir)

        print("Загрузка нового датасета...")
        X_new, y_new = DataLoader.load_data_from_folder(self.new_dataset_dir)

        X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
            X_orig, y_orig, test_size=0.2, random_state=42, stratify=y_orig
        )

        X_train = np.concatenate([X_orig_train, X_new], axis=0)
        y_train = np.concatenate([y_orig_train, y_new], axis=0)

        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        y_train = to_categorical(y_train, num_classes=10)
        y_orig_test = to_categorical(y_orig_test, num_classes=10)

        print(f"Тренировочный набор: {len(X_train)} изображений")
        print(f"Тестовый набор (оригинал): {len(X_orig_test)} изображений")

        return X_train, y_train, X_orig_test, y_orig_test

    def train_conservative(self, X_train, y_train, X_test, y_test, epochs=20):
        model = ModelFactory.create_conservative_model(self.base_model_path)

        original_loss, original_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Оригинальное качество модели: accuracy={original_accuracy:.4f}")

        callbacks = get_training_callbacks(X_test, y_test, original_accuracy, self.results_dir)

        datagen = TrainingUtils.create_ft_augmentation()
        datagen.fit(X_train)

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

    def run_conservative_fine_tuning(self, epochs=20):
        print("🔧 ЗАПУСК КОНСЕРВАТИВНОГО ДООБУЧЕНИЯ")
        print("=" * 60)

        try:
            X_train, y_train, X_test, y_test = self.prepare_conservative_datasets()

            self.fine_tuned_model, self.history = self.train_conservative(
                X_train, y_train, X_test, y_test, epochs=epochs
            )

            original_model = load_model(self.base_model_path)
            original_results = Evaluator.comprehensive_evaluation(original_model, X_test, y_test, "Original")
            fine_tuned_results = Evaluator.comprehensive_evaluation(self.fine_tuned_model, X_test, y_test, "Fine-Tuned")

            Evaluator.save_detailed_results(self.fine_tuned_model, self.results_dir, original_results,
                                            fine_tuned_results)

            print("\n✅ КОНСЕРВАТИВНОЕ ДООБУЧЕНИЕ ЗАВЕРШЕНО!")
            return True

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    config = RecognitionConfig()
    fine_tuner = ConservativeFineTuner(
        config.OCR_MODEL_PATH,
        config.DATASET_DIR,
        config.NEW_DATASET_DIR
    )
    success = fine_tuner.run_conservative_fine_tuning(epochs=20)

    if success:
        print(f"\nМодель сохранена в: {fine_tuner.results_dir}")
        print("Рекомендуется использовать 'meter_recognition_new.keras'")


if __name__ == "__main__":
    main()