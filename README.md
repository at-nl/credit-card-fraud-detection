# Credit Card Fraud Detection
This is an exploration repository of several data science techniques for detecting credit card fraud.

## Data
The dataset used is the [Credit Card Fraud Detection dataset from ULB](https://www.kaggle.com/mlg-ulb/creditcardfraud). This data is available from Kaggle and contains 1 CSV file. 

## Setup
To set up the repository:
1. Clone the repository via HTTPS: [https://github.com/at-nl/credit-card-fraud-detection.git](https://github.com/at-nl/credit-card-fraud-detection.git)
2. Create the environment using the `environment.yml` file by running the following line in command line:

   ```$ conda env create -f environment.yml```

3. Run `main.py` file using the following command:

   ```$ python main.py```

## Methodology
The ULB dataset is famous for being highly imbalanced: 99.83% of the samples are non-fraud, while the rest of the samples are fraud. In this project, methods that deal with this imbalance are used. These methods are:
- Oversampling: Oversampling is defined as adding more copies of the minority class. Oversampling can be a good choice when there is not too much data available. However, oversampling may cause overfitting and poor generalization to the test set.
- Undersampling: Undersamplying is defined as reducing the size of the majority class such that we end up with equal-sized classes in the data. It can be a good choice when there is abundance of data available. However, it may result in information loss and underfitting.
- Synthetic sampling: Synthetic sampling involves generating synthetic samples of the minority class based on the existing samples in the dataset. Synthetic Minority Oversampling Technique (SMOTE) is a popular algorithm to creates sythetic observations of the minority class. ADASYN is another novel adaptive synthetic sampling approach that can be applied to the dataset for this project.

The workflow of the project is as follows:
- The dataset is split into train and test sets.
- The train and test sets are normalized separately.
- The train test is oversampled / undersampled / synthetically sampled.
- Several classifiers are fit into the train test, and evaluated on the test set.
- A grid-search routine to optimize model parameters and select the best performing model is included.

Accuracy is not the best metric to use when evaluating imbalanced datasets as it can be misleading. Metrics that can provide better insight include:
- **Confusion Matrix:** a talbe showing correct predictions and types of incorrect predictions.
- **Precision:** the number of true positives divided by all positive predictions. Precision is also called Positive Predictive Value. It is a measure of a classifier's exactness. Low precision indicates a high number of false positives.
- **Recall:** the number of true positives divided by the number of positive values in the test data. Recall is also called Sensitivity or the True Positive Rate. It is a measure of a classifier's completeness. Low recall indicates a high number of false negatives.
- **F1:** Score: the weighted average of precision and recall.
Since the main objective of the project is to prioritize accuraltely classifying fraud cases, the recall score is the main metric to use for evaluating outcomes.

## References
1. [Methods for Dealing with Imbalanced Data](https://www.kaggle.com/tboyle10/methods-for-dealing-with-imbalanced-data)
