from tensorflow.keras.preprocessing.image import ImageDataGenerator

class TrainingUtils:
    @staticmethod
    def create_base_augmentation():
        return ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            shear_range=0.1,
            horizontal_flip=False
        )

    @staticmethod
    def create_ft_augmentation():
        return ImageDataGenerator(
            rotation_range=8,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.15,
            fill_mode='constant',
            cval=0.0
        )