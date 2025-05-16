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
    raw_features = dataset.loc[:, cols[1:]].values              # Exclude the sample names (index 0) from features
    sample_names = dataset.loc[:, cols[0]].values              # Sample names are in the first column

    mask = ~np.isnan(raw_features).any(axis=1)
    features = raw_features[mask]                               # Drop samples with some missing values
    good_sample_names = sample_names[mask]                          # Keep the sample names corresponding to the features

    bad_sample_names = sample_names[~mask]                      # Sample names with missing values

    scaled_features = StandardScaler().fit_transform(features)  # Standardize the features
    features_pca = PCA(n_components=2).fit_transform(scaled_features)
    return dataset, good_sample_names, bad_sample_names, raw_features, features, scaled_features, features_pca

def detect_outliers(outlier_detector, sample_names, features, alpha=0.05):
    squared_distances = outlier_detector.decision_function(features)
    distances = np.sqrt(np.abs(squared_distances))  # Convert to distances
    print("Max:", np.max(distances), "Min:", np.min(distances))

    p_values = chi2.cdf(distances, df=outlier_detector.n_features_in_)
    is_outlier = p_values > (1 - alpha)
    outlier_names = list(sample_names[is_outlier])
    confidences = p_values
    pct = is_outlier.sum() / len(features) * 100
    return outlier_names, confidences, pct

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
    dataset, good_sample_names, bad_sample_names, raw_features, features, scaled_features, features_pca = fetch_data()
    selected_data = [dataset, raw_features, features, scaled_features, features_pca][2]
    
    outlier_detector = EllipticEnvelope(contamination=0.49, support_fraction=1, random_state=42)
    outlier_detector.fit(features)

    squared_distances = outlier_detector.decision_function(features)
    distances = np.sqrt(np.abs(squared_distances))  # Convert to distances
    print("Max:", np.max(distances), "Min:", np.min(distances))

    p_values = chi2.cdf(distances, df=outlier_detector.n_features_in_)
    is_outlier = p_values > (1 - alpha)
    outlier_names = list(good_sample_names[is_outlier])
    confidences = [(sample_name, p_value) for sample_name, p_value in zip(good_sample_names, p_values)]      # Compile confidence values
    confidences += [(sample_name, 0.0) for sample_name in bad_sample_names]                                  # Add 0.0 confidence for samples with missing values

    pct = is_outlier.sum() / len(features) * 100


    # Detect outliers in the dataset
    # outlier_names, confidences, pct = detect_outliers(outlier_detector, raw_features, alpha)
    print(f"{len(outlier_names)} outliers ({pct:.2f}% of samples):")
    print(outlier_names)

    # extra_plots = [lambda: plt.axhline(y=1-alpha, color='r', linestyle='--', label=f'Selection Threshold: alpha = {alpha}')]
    # plot(outlier_probabilities, "Outlier Probabilities", "Sample Index", "Outlier Probability", extra_plots)
    return outlier_names, confidences

if __name__ == "__main__":
    main()