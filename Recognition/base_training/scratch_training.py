from Recognition.common.config import RecognitionConfig
from Recognition.common.data_loader import DataLoader
from Recognition.common.models import ModelFactory
from Recognition.common.training_utils import TrainingUtils
from Recognition.common.visualization import plot_training_history


def train_from_scratch():
    config = RecognitionConfig()

    images, labels = DataLoader.load_data_from_folder(config.DATASET_DIR)
    X_train, X_val, y_train, y_val = DataLoader.prepare_datasets(images, labels)

    model = ModelFactory.create_baseline_model()
    model.summary()

    datagen = TrainingUtils.create_base_augmentation()
    datagen.fit(X_train)

    history = model.fit(datagen.flow(X_train, y_train, batch_size=config.BASE_BATCH_SIZE),
                        validation_data=(X_val, y_val),
                        epochs=config.BASE_EPOCHS)

    plot_training_history(history)
    loss, acc = model.evaluate(X_val, y_val)
    print(f"Потеря: {loss}, Точность: {acc}")

    model.save(config.BASE_MODEL_SAVE_PATH)
    return model