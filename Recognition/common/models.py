from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam


class ModelFactory:
    @staticmethod
    def create_baseline_model(input_shape=(20, 32, 1), num_classes=10):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    @staticmethod
    def create_conservative_model(base_model_path):
        from tensorflow.keras.models import load_model
        base_model = load_model(base_model_path)

        for layer in base_model.layers[:-4]:
            layer.trainable = False

        for layer in base_model.layers[-4:]:
            layer.trainable = True
            print(f"Разморожен слой: {layer.name}")

        optimizer = Adam(learning_rate=0.0001)
        base_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        return base_model