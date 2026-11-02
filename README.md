# PlantLeafNet – Advanced Plant Disease Detection

## Project Overview

PlantLeafNet is a deep learning-based plant disease detection system that identifies plant diseases from leaf images using transfer learning and model optimization.

The project uses pretrained convolutional neural networks to classify plant leaves into 38 different categories, including healthy and diseased leaves.

## Objectives

- Detect plant diseases from leaf images.
- Compare different pretrained deep learning architectures.
- Improve model performance using transfer learning.
- Optimize the trained model using TensorFlow Lite quantization.
- Develop an interactive web application for disease prediction.

## Dataset

- **Dataset:** PlantVillage
- **Total Images:** 54,305
- **Number of Classes:** 38
- **Image Size:** 256 × 256 pixels
- **Image Format:** RGB

## Models Used

1. EfficientNetB0
2. MobileNetV2

Both models use ImageNet pretrained weights and custom classification heads.

## Model Performance

| Metric | EfficientNetB0 | MobileNetV2 |
|---|---:|---:|
| Test Accuracy | 98.09% | 96.02% |
| Macro F1-score | 0.97 | 0.95 |
| Weighted F1-score | 0.98 | 0.96 |

EfficientNetB0 achieved the best overall performance and was selected for optimization.

## Model Optimization

TensorFlow Lite Dynamic Range Quantization was applied to EfficientNetB0.

| Metric | Original Model | Quantized Model |
|---|---:|---:|
| Accuracy | 98.09% | 98.01% |
| Model Size | 20.12 MB | 4.68 MB |
| Inference Time | 333.25 ms | 86.39 ms |

The quantized model achieved a 76.73% reduction in model size with only a 0.08 percentage-point reduction in accuracy.

## Technologies Used

- Python
- TensorFlow and Keras
- EfficientNetB0
- MobileNetV2
- Albumentations
- NumPy
- Scikit-learn
- Matplotlib and Seaborn
- TensorFlow Lite
- Flask
- HTML and CSS

## Web Application

The Flask application allows users to:

- Upload a plant leaf image.
- Predict the plant disease.
- View the prediction confidence.
- Receive an uncertainty message when confidence is below 70%.

## Project Structure

```text
PlantLeafNet/
│
├── app.py
├── requirements.txt
├── README.md
├── class_names.json
├── efficientnet_plant_disease_quantized.tflite
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

## Installation and Execution

1. Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Navigate to the project folder:

```bash
cd PlantLeafNet
```

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Run the Flask application:

```bash
python app.py
```

5. Open the application in your browser:

```text
http://127.0.0.1:5000
```

## Limitations

- The model was trained using the PlantVillage dataset and may not generalize equally well to real-world field images.
- Visually similar disease symptoms can lead to incorrect predictions.
- Image quality, lighting, and background variations may affect prediction performance.
- The prediction is an AI-based visual assessment and should not replace expert agricultural diagnosis.

## Conclusion

PlantLeafNet demonstrates the use of transfer learning for automated plant disease classification. EfficientNetB0 achieved the highest test accuracy, while TensorFlow Lite quantization significantly reduced model size and inference time. The Flask application provides an interactive interface for testing the optimized model.