# Plant Disease Detection using Various CNN Models

## Overview
This project aims to compare the performance of different Convolutional Neural Network (CNN) models for the task of plant disease detection. The models used in this project are ResNet50, MobileNetV2, and VGG15. The accuracy achieved for each model is as follows: ResNet50 - 96.16%, MobileNetV2 - 92.53%, VGG15 - 80.28%.

## Objective
The primary objective of this project is to evaluate the effectiveness of different CNN architectures in detecting plant leaf diseases. By comparing the performance of ResNet50, MobileNetV2, and VGG15 models, we aim to identify the most suitable model for this specific task.

## Methodology
1. **Data Collection**: Gathered a dataset of plant leaf images containing healthy leaves as well as leaves affected by various diseases.
2. **Preprocessing**: Preprocessed the images by resizing, normalizing, and augmenting the data to ensure uniformity and enhance model performance.
3. **Model Training**: Trained ResNet50, MobileNetV2, and VGG15 models on the preprocessed dataset.
4. **Fine-tuning**: Tailored the classification layers of each model to better suit the plant leaf disease detection task.
5. **Evaluation**: Evaluated the performance of each model using metrics such as accuracy, precision, recall, and F1-score.
6. **Comparison**: Compared the accuracy and performance of ResNet50, MobileNetV2, and VGG15 models to determine the most effective model for plant disease detection.

## Results
- ResNet50: Accuracy - 96.16%
- MobileNetV2: Accuracy - 92.53%
- VGG15: Accuracy - 80.28%

## Conclusion
Based on the results, ResNet50 outperforms MobileNetV2 and VGG15 models in terms of accuracy for plant disease detection. However, further analysis is required to understand the trade-offs between model accuracy, computational efficiency, and deployment considerations.

## Future Work
- Experiment with other CNN architectures and hyperparameters to potentially improve performance.
- Explore ensemble methods to combine the strengths of multiple models for enhanced disease detection accuracy.
- Investigate the generalization of the trained models on unseen datasets and real-world scenarios.

[Link to Dataset used](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
