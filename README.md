[West_Nile_Virus_Project_Documentation.txt](https://github.com/user-attachments/files/24553063/West_Nile_Virus_Project_Documentation.txt)
West Nile Virus Outbreak Prediction Documentation
This document is the official comprehensive documentation for the West Nile Virus outbreak prediction project. It is authored by the project developer and details every step from data ingestion to model evaluation.
1. Project Objective
The objective of this project is to build a machine learning model to predict the presence of West Nile Virus (WnvPresent) in mosquito trap records using weather, trap, and spray datasets. The solution includes advanced feature engineering grounded in biological reasoning and data science principles.
2. Repository Structure Overview
The repository follows a modular structure for clarity and collaboration:
� data/raw: original CSV data files.
� data/processed: cleaned data ready for modeling.
� src: Python modules for preprocessing, feature engineering, and modeling.
� notebooks: exploratory and modeling notebooks.
� reports: generated visualization outputs.
3. Installation and Setup
To run this project locally, create a Python virtual environment and install dependencies from requirements.txt using pip. Ensure that python-docx is also installed to allow generation of Word documentation.
4. Data Sources and Structure
This project integrates three primary datasets:
� Train.csv: Contains mosquito trap collection records, coordinates, species, and virus presence labels.
� Weather.csv: Daily weather records from two stations with complex missing values.
� Spray.csv: Insecticide spray application logs with spatial and temporal fields.
5. Train Dataset Preprocessing
Data cleaning included date parsing with extraction of year, month, week, and day-of-year components. Duplicate records were handled based on trap, date, and species keys before merging with weather and spray data.
6. Weather Dataset Comprehensive Cleaning
The weather dataset required the most extensive cleaning:
- Converting string flags like 'M' (missing) and 'T' (trace) to numeric values.
- Converting Fahrenheit temperature readings to Celsius for consistency.
- Estimating missing sunrise/sunset values for Station 2 using astronomical calculations adjusted for local standard time.
For certain fields, station-specific imputation methods were applied to preserve meaningful weather trends.
7. Handling Missing Station 2 Weather Values
Station 2 weather values were missing for sunrise, sunset, and some other fields. Rather than dropping these, I generated estimates using reliable astronomical calculations based on latitude and longitude and filled missing values accordingly to preserve data completeness.
8. Exploratory Data Analysis (EDA)
EDA was performed to understand the seasonal patterns in weather and mosquito activity, correlation between weather events, and preliminary data behavior. Visualizations such as histograms, time-series plots, and correlation heatmaps were generated for insight.
9. Feature Engineering
Advanced feature engineering included:
� Temporal lags (e.g., 14-day rolling averages for temperature and precipitation).
� Cumulative heat-related metrics.
� Relative humidity calculations.
� Daylight change metrics.
� Weather events and intensity encoding.
� Spray proximity and intensity features.
10. Data Merging Process
Weather and spray features were merged into the train dataset using appropriate join keys and spatial-temporal logic to ensure accurate alignment of records.
11. Correlation and Feature Redundancy
A full correlation analysis helped identify and drop highly correlated redundant features to improve model stability and reduce noise.
12. Dimensionality Reduction (PCA)
Principal Component Analysis was used to reduce dimensionality after scaling. 25 components were selected based on explained variance and plotted for review.
13. Modeling Approach
An XGBoost classifier was selected for its robustness to nonlinear interactions and class imbalance. The model was trained with careful parameter selection and evaluated using ROC-AUC and Precision-Recall metrics.
14. Train-Test Splitting
Time-aware splits by year prevented temporal leakage: training on earlier years and testing on later data to ensure valid evaluation.
15. Results Summary
The final model showed significant improvements over baseline with strong predictive performance validated by both ROC-AUC and PR-AUC curves.
16. Project Dependencies
Key packages used include pandas, numpy, scikit-learn, xgboost, seaborn, geopy, and python-docx for documentation generation.
17. Running the Project
Execute `python main.py` from the project root to run the full pipeline: cleaning, feature engineering, PCA, model training, and evaluation.
18. Future Enhancements
Future improvements include hyperparameter tuning, alternate model experiments, and automated reporting dashboards.
