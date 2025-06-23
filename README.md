# Measurement Outlier Detector

This repository provides tools and scripts for detecting outliers in measurement datasets using machine learning ensemble methods.

## Repository Structure

-`Measurement Outlier Detector.py`

Python script for detecting measurement outliers.

You can specify which variables make up which measurements, in case the current configuration is not entirely correct.

The models generated and reused by this code are in the models folder.



Then just run the code and a final "measurement_outlier_confidences.csv" with outlier confidence values should appear.

If you want to filter which of these measurements are outliers per sample, simply select all cells with value < alpha threshold (self-selected alpha value).

So we currently have not distinguished measurement outliers yet.





-`Sample Outlier Detector ensemble.ipynb`

The main ensemble for detecting sample outliers.

In principle, simply run all the cells one after another. 



We built it such that the final cell will run all the code.

This is also where the alpha threshold value can be chosen to determine the outliers.





-`measurement_outlier_confidences.csv`

Output CSV file containing measurement outlier confidence scores.

-`models.zip`

Pre-trained models or model artifacts used by the detector.

## 

## Authors

- Emiel Nagel
- Patricia Priscorniță 
- Ramireddy, A.R. (Abhinav)
- Nicolas Sanmartin de Miranda

---

*This repository is intended for research and educational purposes.*
