# Disease Prediction System (Diabetes & Heart Disease)

A machine learning project that predicts the likelihood of **diabetes** and **heart disease** from patient health data, with a Streamlit web interface.

**Author:** Soumya, BTech CSE, KIIT University


> Educational project. Predictions are not a medical diagnosis.

## Datasets

| Disease       | Dataset                                      | Source                     | Download                                                             |
| ------------- | -------------------------------------------- | -------------------------- | -------------------------------------------------------------------- |
| Diabetes      | PIMA Indians Diabetes (768 rows, 8 features) | UCI ML Repository / Kaggle | https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database |
| Heart disease | Heart Disease (Cleveland-based, 13 features) | UCI ML Repository / Kaggle | https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset    |

Datasets are not stored in this repo. Put them in a `data/` folder:

- `data/diabetes.csv` is downloaded automatically by `train.py` if missing.
- `data/heart.csv` must be downloaded from Kaggle (free account needed).

## Tech Stack

Python, Pandas, NumPy, scikit-learn, Matplotlib, Streamlit

## Project Structure

```
.
├── common.py          # shared feature lists + custom outlier-clipping transformer
├── train.py           # preprocessing, feature selection, training, evaluation
├── app.py             # Streamlit UI
├── requirements.txt
├── data/              # datasets (not committed)
├── models/            # saved models (created by train.py)
└── outputs/           # result tables and plots (created by train.py)
```

## How to Run

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Methodology

1. **Define problem:** binary classification, i.e. does the patient have diabetes / heart disease.
2. **Collect data:** PIMA diabetes and Heart Disease datasets.
3. **Preprocess:**
   - In the PIMA data, zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI are impossible values and are treated as missing.
   - Missing values are filled with the median.
   - Outliers in continuous columns are clipped using the IQR rule.
   - Duplicate rows in the heart dataset are removed so they cannot leak between train and test sets.
4. **Feature selection and normalisation:** `SelectKBest` (ANOVA F-test) with `k` tuned by cross-validation, and `StandardScaler`.
5. **Model patterns:** correlation heatmaps and class balance checks (`outputs/*_correlation.png`).
6. **Train models:** Logistic Regression, KNN, SVM, Random Forest and Gradient Boosting, tuned with `GridSearchCV`. All preprocessing sits inside one scikit-learn `Pipeline`, so it is fitted only on training folds (no data leakage).
7. **Evaluate performance:** accuracy, precision, recall, F1 and ROC-AUC on a held-out 20% test set, plus confusion matrices and ROC curves. Recall matters most in medical screening because a missed case is costlier than a false alarm.
8. **Cross-validation:** stratified 5-fold CV on the training set. The best model is chosen by CV ROC-AUC only, so the test set is never used for model selection.
9. **Develop UI:** Streamlit app with a form per disease, estimated probability and model information.
10. **Ethics and limitations:** see below.

## Results

Fill these from `outputs/diabetes_results.csv` and `outputs/heart_results.csv` after running `train.py`.

| Disease  | Best model | Accuracy | Recall | F1    | ROC-AUC |
| -------- | ---------- | -------- | ------ | ----- | ------- |
| Diabetes | _add_      | _add_    | _add_  | _add_ | _add_   |
| Heart    | _add_      | _add_    | _add_  | _add_ | _add_   |

Plots are saved in `outputs/`: correlation matrices, confusion matrices and ROC curves.

## Ethics and Limitations

- **Not a diagnostic tool.** The app is a learning project and must not replace a doctor.
- **Small, dated data.** PIMA has 768 patients and Heart has about 300 unique patients. Results may not hold on other populations.
- **Population bias.** PIMA contains only women of Pima Indian heritage aged 21+. The model should not be assumed to work for men, other ethnic groups or children.
- **Missing values in PIMA** (e.g. insulin) are imputed, which adds uncertainty.
- **Class imbalance and threshold.** Predictions use a 0.5 threshold. A real screening tool would tune this to favour recall.
- **Privacy.** No patient data is stored by the app. Real deployments need consent, data protection and clinical validation.
- **Fairness.** Performance was not audited across subgroups.

## References

- Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. Proc. Symp. Comput. Appl. Med. Care, 261-265.
- Detrano, R. et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. American Journal of Cardiology, 64(5), 304-310.
- UCI Machine Learning Repository: https://archive.ics.uci.edu
- scikit-learn: Pedregosa et al. (2011), JMLR 12, 2825-2830.


