import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Потеря на обучении')
    plt.plot(history.history['val_loss'], label='Потеря на валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Потеря')
    plt.legend()
    plt.title('График потерь')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Точность на обучении')
    plt.plot(history.history['val_accuracy'], label='Точность на валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.title('График точности')

    plt.show()