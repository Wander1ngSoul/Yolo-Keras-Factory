import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import matplotlib.gridspec as gridspec

from Recognition.common.config import RecognitionConfig

load_dotenv()


def preprocess_original(image):
    image = cv2.resize(image, (32, 20))
    return image


def test_single_image(image_path, model):
    original_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_color = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if original_gray is None:
        print(f"Ошибка загрузки изображения: {image_path}")
        return None

    processed_img = preprocess_original(original_gray.copy())

    img_normalized = processed_img.astype("float32") / 255.0
    img_input = np.expand_dims(img_normalized, axis=-1)
    img_input = np.expand_dims(img_input, axis=0)

    prediction = model.predict(img_input, verbose=0)[0]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class] * 100

    results = {
        'processed_image': processed_img,
        'prediction': prediction,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'top3': sorted([(i, p * 100) for i, p in enumerate(prediction)],
                       key=lambda x: x[1], reverse=True)[:3]
    }

    print(f"Изображение: {os.path.basename(image_path)}")
    print(f"Распознанная цифра: {predicted_class}")
    print(f"Уверенность: {confidence:.1f}%")
    print(f"Топ-3: {results['top3']}")
    print("-" * 40)

    return results, original_color


def plot_results(results, original_color, image_name):
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f'Распознавание цифры: {image_name}', fontsize=14, fontweight='bold')

    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], hspace=0.4, wspace=0.3)

    ax_original = plt.subplot(gs[0, 0])
    ax_original.imshow(cv2.cvtColor(original_color, cv2.COLOR_BGR2RGB))
    ax_original.set_title('Оригинальное изображение', fontsize=12, pad=10)
    ax_original.axis('off')

    ax_processed = plt.subplot(gs[0, 1])
    ax_processed.imshow(results['processed_image'], cmap='gray')
    ax_processed.set_title('После обработки (32x20)', fontsize=12, pad=10)
    ax_processed.axis('off')

    ax_confidence = plt.subplot(gs[1, 0])
    digits = range(10)
    probabilities = results['prediction'] * 100

    colors = ['red' if i != results['predicted_class'] else 'green' for i in range(10)]
    bars = ax_confidence.bar(digits, probabilities, color=colors, alpha=0.7, edgecolor='black')

    ax_confidence.set_xlabel('Цифра', fontsize=11)
    ax_confidence.set_ylabel('Уверенность (%)', fontsize=11)
    ax_confidence.set_title('Уверенность распознавания', fontsize=12, pad=10)
    ax_confidence.set_ylim(0, 110)
    ax_confidence.grid(True, alpha=0.3, axis='y')

    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 5:
            ax_confidence.text(bar.get_x() + bar.get_width() / 2., height + 2,
                               f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    ax_top3 = plt.subplot(gs[1, 1])
    top3_digits = [x[0] for x in results['top3']]
    top3_probs = [x[1] for x in results['top3']]

    colors_top3 = ['#FF9999', '#66B2FF', '#99FF99']
    bars = ax_top3.bar(range(3), top3_probs, color=colors_top3, alpha=0.8, edgecolor='black')

    ax_top3.set_xticks(range(3))
    ax_top3.set_xticklabels([f'Цифра {d}' for d in top3_digits], fontsize=10)
    ax_top3.set_ylabel('Вероятность (%)', fontsize=11)
    ax_top3.set_title('Топ-3 предсказания', fontsize=12, pad=10)
    ax_top3.set_ylim(0, 110)
    ax_top3.grid(True, alpha=0.3, axis='y')

    for i, (bar, prob) in enumerate(zip(bars, top3_probs)):
        ax_top3.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 2,
                     f'{prob:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()


def main():
    config = RecognitionConfig()

    model_path = config.OCR_MODEL_PATH
    test_image_path = os.getenv('TEST_IMG_PATH')

    if not test_image_path:
        print("❌ Ошибка: Не задан путь к тестовому изображению в .env")
        print("Добавьте TEST_IMG_PATH=ваш/путь/к/изображению.jpg в файл .env")
        return

    print(f"Загрузка модели: {model_path}")

    try:
        model = load_model(model_path)
        print("✅ Модель успешно загружена")
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return

    print(f"Тестирование изображения: {test_image_path}")
    print("=" * 50)

    results, original_color = test_single_image(test_image_path, model)

    if results:
        image_name = os.path.basename(test_image_path)
        plot_results(results, original_color, image_name)

        print("\n" + "=" * 50)
        print("ИТОГИ РАСПОЗНАВАНИЯ:")
        print(f"Цифра: {results['predicted_class']}")
        print(f"Уверенность: {results['confidence']:.1f}%")

        if results['confidence'] > 80:
            print("✅ Высокая уверенность распознавания")
        elif results['confidence'] > 60:
            print("⚠️  Средняя уверенность распознавания")
        else:
            print("❌ Низкая уверенность распознавания")
        print("=" * 50)


if __name__ == "__main__":
    main()