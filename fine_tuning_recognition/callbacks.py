import os
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

class QualityPreservationCallback(Callback):
    def __init__(self, test_data, original_accuracy, results_dir):
        super().__init__()
        self.X_test, self.y_test = test_data
        self.original_accuracy = original_accuracy
        self.best_accuracy = original_accuracy
        self.results_dir = results_dir

    def on_epoch_end(self, epoch, logs=None):
        current_accuracy = logs.get('val_accuracy', 0)
        if current_accuracy >= self.original_accuracy - 0.02:
            if current_accuracy > self.best_accuracy:
                self.best_accuracy = current_accuracy
                self.model.save(os.path.join(self.results_dir, 'meter_recognition_new.keras'))
                print(f"✓ Сохранена модель с accuracy: {current_accuracy:.4f}")

def get_training_callbacks(X_test, y_test, original_accuracy, results_dir):
    return [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        QualityPreservationCallback((X_test, y_test), original_accuracy, results_dir),
        ModelCheckpoint(
            os.path.join(results_dir, 'checkpoint_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]