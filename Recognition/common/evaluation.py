import numpy as np
import json
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import os


class Evaluator:
    @staticmethod
    def comprehensive_evaluation(model, X_test, y_test, model_name="Model"):
        results = {}

        if len(X_test.shape) == 3:
            X_test = X_test.reshape(-1, 20, 32, 1)

        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        results['accuracy'] = np.mean(y_pred_classes == y_true_classes)
        results['report'] = classification_report(y_true_classes, y_pred_classes, output_dict=True)
        results['confusion_matrix'] = confusion_matrix(y_true_classes, y_pred_classes)

        robustness_tests = [
            ('Original', lambda x: x),
            ('Gaussian Noise', lambda x: np.clip(x + np.random.normal(0, 0.05, x.shape), 0, 1)),
            ('Blur', lambda x: cv2.GaussianBlur(x, (3, 3), 0) if len(x.shape) == 2 else x),
            ('Darken', lambda x: np.clip(x * 0.8, 0, 1)),
            ('Brighten', lambda x: np.clip(x * 1.2, 0, 1)),
        ]

        robustness_results = {}
        for name, transform in robustness_tests:
            try:
                X_transformed = np.array([transform(x.copy()) for x in X_test])
                if len(X_transformed.shape) == 3:
                    X_transformed = X_transformed.reshape(-1, 20, 32, 1)
                loss, acc = model.evaluate(X_transformed, y_test, verbose=0)
                robustness_results[name] = acc
            except Exception as e:
                print(f"Ошибка в тесте {name}: {e}")
                robustness_results[name] = 0.0

        results['robustness'] = robustness_results
        return results

    @staticmethod
    def save_detailed_results(fine_tuned_model, results_dir, original_results, fine_tuned_results):
        fine_tuned_model.save(os.path.join(results_dir, 'final_fine_tuned_model.keras'))

        results = {
            'original_accuracy': float(original_results['accuracy']),
            'fine_tuned_accuracy': float(fine_tuned_results['accuracy']),
            'improvement': float(fine_tuned_results['accuracy'] - original_results['accuracy']),
            'robustness_comparison': {
                'original': original_results['robustness'],
                'fine_tuned': fine_tuned_results['robustness']
            }
        }

        with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
            json.dump(results, f, indent=2)

        with open(os.path.join(results_dir, 'detailed_report.txt'), 'w', encoding='utf-8') as f:
            f.write("ДЕТАЛЬНЫЙ ОТЧЕТ КОНСЕРВАТИВНОГО ДООБУЧЕНИЯ\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Оригинальная точность: {original_results['accuracy']:.4f}\n")
            f.write(f"Дообученная точность: {fine_tuned_results['accuracy']:.4f}\n")
            f.write(f"Изменение: {fine_tuned_results['accuracy'] - original_results['accuracy']:+.4f}\n\n")

            f.write("ТЕСТ НА УСТОЙЧИВОСТЬ:\n")
            for test_name in original_results['robustness']:
                orig_acc = original_results['robustness'][test_name]
                ft_acc = fine_tuned_results['robustness'][test_name]
                diff = ft_acc - orig_acc
                f.write(f"{test_name:15}: {orig_acc:.3f} -> {ft_acc:.3f} ({diff:+.3f})\n")