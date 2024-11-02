# Accurate-Seizure-Detection-from-EEG-Signals-using-Hybrid-Machine-Learning-Models
Project Overview
This project explores a hybrid approach to detecting seizure phases from EEG signals using both traditional machine learning models and deep learning architectures. By leveraging the Bonn EEG dataset, the project aims to classify seizure activity into distinct phases (ictal, pre-ictal, and interictal) and improve accuracy for real-time prediction and management of epilepsy. Various supervised learning algorithms, as well as Long Short-Term Memory (LSTM) networks, were applied to uncover patterns in EEG data and improve seizure detection.

Dataset
The Bonn EEG dataset, collected by the University of Bonn's Department of Epileptology, serves as the primary dataset for this project. It’s widely used in seizure detection research for its rich and structured data on brain activity.

Dataset Structure
Samples: The dataset contains five subsets (A–E), each with 100 single-channel EEG recordings, sampled at 173.6 Hz over 23.6 seconds.
Classes:
Set A & B: Healthy individuals (A - eyes open, B - eyes closed)
Set C & D: Seizure-free intervals in epileptic patients
Set E: Seizure activity in epileptic patients
Labeling: Enables multi-class classification, supporting the analysis across ictal, pre-ictal, and interictal states.
Methodology
This project involves several key steps, from data preprocessing to model training and evaluation.

1. Data Preprocessing
Exploratory Data Analysis (EDA): Assessed the dataset structure, distributions, and any outliers.
Data Normalization: Applied standard scaling to ensure features have zero mean and unit variance, improving model convergence.
Label Encoding: Categorical labels were encoded for model training, converting seizure phases to numerical values.
Handling Class Imbalance: Addressed imbalances in seizure and non-seizure events to enhance model robustness.
2. Model Architecture and Selection
Supervised Machine Learning Models
Logistic Regression: Provides a baseline with interpretability but limited in capturing non-linear patterns.
Random Forest: An ensemble model capable of handling complex data and providing interpretability through feature importance.
Support Vector Machine (SVM): Effective for high-dimensional EEG data, capable of distinguishing seizure and non-seizure events with appropriate kernels.
Artificial Neural Network (ANN): A multi-layer perceptron that captures non-linear relationships and improves accuracy compared to simpler models.
Deep Learning Model
LSTM Network: Specifically designed for sequential data like EEG signals, the LSTM models temporal dependencies crucial for distinguishing seizure phases.
3. Hybrid Model with CNN and LSTM
Architecture: Combines convolutional layers for spatial feature extraction, a Bidirectional LSTM layer for temporal sequence learning, and an attention mechanism to focus on the most relevant EEG segments.
Training and Validation: The model was trained over 100 epochs using adaptive learning rate adjustments and early stopping to optimize performance.
Results
Below are the performance metrics (Accuracy, Precision, Recall, F1 Score) for each model, showcasing their effectiveness in detecting seizure phases.

## Results

| Model                          | Accuracy | Precision | Recall | F1 Score |
|--------------------------------|----------|-----------|--------|----------|
| **Logistic Regression**        | 71.57%   | 34.12%    | 43.66% | 38.30%   |
| **Random Forest**              | 97.22%   | 97.40%    | 88.60% | 92.79%   |
| **Support Vector Machine (SVM)** | 97.87%   | 92.98%    | 96.77% | 94.84%   |
| **Artificial Neural Network (ANN)** | 97.70%   | 96.19%    | 92.26% | 94.18%   |
| **Hybrid CNN-LSTM with Attention** | ~90%  | Balanced across metrics | High interpretability | High temporal accuracy |

Discussion
Best Performing Models: The Hybrid CNN-LSTM with Attention model achieved a balanced performance with high generalization ability, while SVM and ANN showed excellent results among traditional models.
Insights: The LSTM's ability to capture temporal dependencies, combined with the attention mechanism, allowed the hybrid model to focus on key EEG segments, enhancing seizure detection.
Challenges: Logistic regression struggled with non-linear data, demonstrating the need for more sophisticated models in this context.
Conclusion
The combination of CNN, LSTM, and attention mechanism has proven effective in detecting seizure events from EEG signals, with an emphasis on the importance of temporal dependencies for accurate classification. This approach holds potential for real-world applications in epilepsy care by enabling real-time seizure prediction and intervention.

Future Directions
To further enhance the seizure detection system, several improvements and extensions are proposed:

Advanced Architectures: Integrate Transformer models for capturing long-range dependencies and attention in EEG data.
Data Augmentation: Use Generative Adversarial Networks (GANs) to generate synthetic EEG data and address class imbalance.
Multi-modal Integration: Combine EEG data with other physiological data, such as heart rate and oxygen levels, for a more comprehensive seizure prediction system.
Model Optimization for Real-Time Applications: Explore techniques like pruning and quantization to deploy the model on wearable devices for real-time monitoring.

References
Shoeb, A. H., & Guttag, J. V. (2010). Application of machine learning to epileptic seizure detection.
Acharya, U. R., et al. (2013). Automated diagnosis of epilepsy using EEG and machine learning.
Hosseini, M. P., & Pompili, D. (2020). Real-time seizure detection in EEG using multi-stage SVM and phase-space analysis.
