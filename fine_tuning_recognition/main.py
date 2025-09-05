import os
import numpy as np
from datetime import datetime
from fine_tuning_recognition import (
    FineTuningConfig, DataLoader, ModelUtils, Trainer, Evaluator
)
from tensorflow.keras.models import load_model

class ConservativeFineTuner:
    def __init__(self, base_model_path, original_dataset_dir, new_dataset_dir):
        self.base_model_path = base_model_path
        self.original_dataset_dir = original_dataset_dir
        self.new_dataset_dir = new_dataset_dir
        self.base_model = None
        self.fine_tuned_model = None
        self.history = None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = os.path.join(FineTuningConfig.RESULTS_BASE_DIR, f'conservative_fine_tuning_{timestamp}')
        os.makedirs(self.results_dir, exist_ok=True)

        print(f"Результаты будут сохранены в: {self.results_dir}")

    def run_conservative_fine_tuning(self, epochs=20):
        print("🔧 ЗАПУСК КОНСЕРВАТИВНОГО ДООБУЧЕНИЯ")
        print("=" * 60)

        try:
            X_train, y_train, X_test, y_test = DataLoader.prepare_conservative_datasets(
                self.original_dataset_dir, self.new_dataset_dir
            )

            model = ModelUtils.create_conservative_model(self.base_model_path)
            trainer = Trainer(self.results_dir)
            self.fine_tuned_model, self.history = trainer.train_conservative(
                model, X_train, y_train, X_test, y_test, epochs=epochs
            )

            original_model = load_model(self.base_model_path)
            original_results = Evaluator.comprehensive_evaluation(original_model, X_test, y_test, "Original")
            fine_tuned_results = Evaluator.comprehensive_evaluation(self.fine_tuned_model, X_test, y_test, "Fine-Tuned")

            Evaluator.save_detailed_results(self.fine_tuned_model, self.results_dir, original_results, fine_tuned_results)

            print("\n✅ КОНСЕРВАТИВНОЕ ДООБУЧЕНИЕ ЗАВЕРШЕНО!")
            return True

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    config = FineTuningConfig()
    fine_tuner = ConservativeFineTuner(
        config.BASE_MODEL_PATH,
        config.ORIGINAL_DATASET_DIR,
        config.NEW_DATASET_DIR
    )
    success = fine_tuner.run_conservative_fine_tuning(epochs=20)

    if success:
        print(f"\nМодель сохранена в: {fine_tuner.results_dir}")
        print("Рекомендуется использовать 'meter_recognition_new.keras'")

if __name__ == "__main__":
    main()