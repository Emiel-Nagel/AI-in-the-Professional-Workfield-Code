import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from scipy.stats import chi2

import matplotlib.pyplot as plt
import seaborn as sns

def fetch_data():
    # Load the dataset
    dataset = pd.read_csv('Code_Emiel/Dataset/refData_obf.csv')

    cols = dataset.columns
    features = dataset.loc[:, cols[1:]].values              # Exclude the sample names (index 0) from features
    mask = ~np.isnan(features).any(axis=1)
    features = features[mask]                               # Drop samples with some missing values
    scaled_features = StandardScaler().fit_transform(features)  # Standardize the features
    features_pca = PCA(n_components=2).fit_transform(scaled_features)
    return dataset, features, scaled_features, features_pca

def detect_outliers(outlier_detector, features, alpha=0.05):
    # if type(features[0]) == str:
    #     features = [features]
    squared_distances = outlier_detector.decision_function(features)
    distances = np.sqrt(np.abs(squared_distances))  # Convert to distances
    print("Max:", np.max(distances), "Min:", np.min(distances))
    # log_distances = np.log(distances)

    order = np.argsort(distances)
    rank = np.empty_like(distances)
    rank[order] = np.arange(len(distances))
    print(rank)
    p_emp = rank / len(distances)  # Normalize ranks to [0, 1]

    outlier_probabilities = chi2.cdf(distances, df=outlier_detector.n_features_in_)
    outliers = np.where(outlier_probabilities > 1 - alpha)[0]
    percentage_outliers = (outliers.size / len(features)) * 100
    return outlier_probabilities, outliers, percentage_outliers

def plot(data, title, xlabel, ylabel, extra_plots=[]):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=np.arange(len(data)), y=data)
    for extra_plot in extra_plots:
        extra_plot()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    alpha = 0.05
    selected_data_type = 1
    dataset, features, scaled_features, features_pca = fetch_data()
    selected_data = [dataset, features, scaled_features, features_pca][selected_data_type]
    
    outlier_detector = EllipticEnvelope(contamination=0.49, support_fraction=1, random_state=42)
    outlier_detector.fit(selected_data)

    # Detect outliers in the dataset
    outlier_probabilities, outliers, percentage_outliers = detect_outliers(outlier_detector, selected_data, alpha)
    print(f"{len(outliers)} outliers detected from alpha = {alpha}:")
    print(outliers)
    print(f"Percentage outliers in the dataset: {percentage_outliers:.2f}%")

    extra_plots = [lambda: plt.axhline(y=1-alpha, color='r', linestyle='--', label=f'Selection Threshold: alpha = {alpha}')]
    plot(outlier_probabilities, "Outlier Probabilities", "Sample Index", "Outlier Probability", extra_plots)
    return "Hello, World!"