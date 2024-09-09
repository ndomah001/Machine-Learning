# Machine Learning
## ❓ What is Machine Learning
**Machine learning (ML)** is a subset of artificial intelligence (AI) that involves training algorithms to identify patterns and make decisions based on data. It allows computers to learn from experience without being explicitly programmed for specific tasks. In essence, the system improves its performance over time as it is exposed to more data.

In bioinformatics, ML is used to extract meaningful insights from large and complex biological datasets, where traditional methods might struggle. Machine learning is revolutionizing bioinformatics by enabling automated analysis of complex biological data, allowing researchers to make more accurate predictions and discoveries in genomics, transcriptomics, proteomics, and other fields.

## 📝 Introduction
Promoter gene sequences play a critical role in the regulation of gene expression in *E. Coli* and other organisms. Identifying promoter sequences within DNA is a challenging task that requires precise classification. In this project, I explore machine learning techniques to classify *E. Coli* DNA sequences into promoter or non-promoter categories. The goal is to develop models that can accurately predict whether a given DNA sequence has promoter properties based on the sequence's encoded features.

The [dataset](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq) used consists of 106 instances of DNA sequences from *E. Coli*. These sequences are represented in categorical form, requiring conversion into numerical data before being fed into machine learning models. Each sequence is labeled as either a promoter (positive class) or non-promoter (negative class).

**Key Dataset Characteristics**:
- **Instances**: 106
- **Features**: Categorical representation of DNA sequences
- **Target**: Binary classification (promoter vs. non-promoter)

## 📜 Methodology
To classify the DNA sequences, I employed a variety of machine learning algorithms, each evaluated based on accuracy and other performance metrics. The models used in this study include:

- **K-Nearest Neighbors (KNN)**: A distance-based algorithm that classifies data points based on their proximity to labeled instances.
- **Gaussian Process**: A probabilistic model that applies kernel functions to capture relationships in the data.
- **Decision Tree**: A simple model that splits the data into subsets based on feature values to create a tree-like structure.
- **Random Forest**: An ensemble method using multiple decision trees to improve classification robustness.
- **Neural Network**: A multilayer perceptron model (MLPClassifier) that captures non-linear patterns in the data.
- **AdaBoost**: An ensemble method that combines multiple weak classifiers to improve overall performance.
- **Naive Bayes**: A probabilistic model based on the assumption of conditional independence between features.
- **Support Vector Machine (SVM)**: A robust classifier tested with multiple kernel functions (linear, RBF, sigmoid) to find the best hyperplane for separating the classes.

Each model was evaluated using 10-fold cross-validation to ensure reliable performance estimates. After training, I generated classification reports to evaluate precision, recall, F1-score, and accuracy for each model.

## 📈 Results
The performance of the models was assessed using accuracy, precision, recall, and F1-scores. Below is a summary of the cross-validation accuracy for each model:

| Model                  | Accuracy (Mean ± Std) |
|-------------------------|----------------------|
| K-Nearest Neighbors      | 0.90 ± 0.05          |
| Gaussian Process         | 0.85 ± 0.07          |
| Decision Tree            | 0.88 ± 0.06          |
| Random Forest            | 0.92 ± 0.04          |
| Neural Network (MLP)     | 0.89 ± 0.06          |
| AdaBoost                 | 0.86 ± 0.05          |
| Naive Bayes              | 0.84 ± 0.07          |
| SVM (Linear Kernel)      | 0.91 ± 0.04          |
| SVM (RBF Kernel)         | 0.93 ± 0.03          |
| SVM (Sigmoid Kernel)     | 0.87 ± 0.05          |

From these results, **Random Forest** and **SVM with RBF kernel** emerged as the best-performing models, with accuracies of 92% and 93%, respectively. **K-Nearest Neighbors** and **SVM with a linear kernel** also performed well, achieving accuracies around 90%.

Additionally, the classification reports for the top-performing models provided detailed insights into how well each model handled promoter and non-promoter classifications. For example, the SVM (RBF) achieved a high F1-score for both classes, indicating a balanced performance across precision and recall.

## 💬 Discussion
The results suggest that ensemble methods like Random Forest, and kernel-based approaches like SVM with an RBF kernel, are well-suited for this classification task. The robustness of these models lies in their ability to capture complex patterns in the DNA sequence data, which may not be apparent using simpler models like Naive Bayes or Decision Tree.

- **Random Forest** benefits from combining multiple decision trees to reduce overfitting, making it a strong performer for this dataset.
- **SVM with an RBF kernel** can efficiently handle non-linear boundaries, making it a particularly effective model for biological sequence data, which often exhibits intricate patterns.
- **K-Nearest Neighbors** also performed well, likely due to the clear separation between promoter and non-promoter sequences in the feature space.

While models like **Neural Networks** and **AdaBoost** showed reasonable accuracy, their performance was slightly lower than Random Forest and SVM, potentially due to the small size of the dataset, which may not have provided enough data for these models to generalize effectively.

## ✅ Conclusion
In this project, I explored a range of machine learning models to classify *E. Coli* promoter gene sequences. The Random Forest and SVM (RBF kernel) models demonstrated the highest accuracy and balanced performance across all metrics. These models provide a promising approach for future applications in genomic classification tasks.

Moving forward, additional steps such as hyperparameter tuning, increasing the dataset size, and exploring other advanced models (e.g., deep learning methods) could further improve classification performance.
