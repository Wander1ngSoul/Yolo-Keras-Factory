import os
from dotenv import load_dotenv
from Recognition.common.data_loader import DataLoader
from Recognition.common.models import ModelFactory
from Recognition.common.training_utils import TrainingUtils
from Recognition.common.visualization import plot_training_history

load_dotenv()


def main():
    dataset_dir = os.getenv('DATASET_DIR_PATH_ORIGIN')

    images, labels = DataLoader.load_data_from_folder(dataset_dir)

    X_train, X_val, y_train, y_val = DataLoader.prepare_datasets(images, labels)

    model = ModelFactory.create_baseline_model()

    datagen = TrainingUtils.create_base_augmentation()
    datagen.fit(X_train)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=75
    )

    plot_training_history(history)

    loss, acc = model.evaluate(X_val, y_val)
    print("Потеря на валидации:", loss)
    print("Точность на валидации:", acc)

    model.save("meter_recognition.keras")


if __name__ == "__main__":
    main()