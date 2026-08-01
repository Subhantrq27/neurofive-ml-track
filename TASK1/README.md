# neurofive-ml-track

My work for the Neurofive Solutions ML track.

## Task 1: Data Science Toolkit Setup & First EDA

- `titanic_eda.ipynb` — loads the Titanic dataset with pandas, inspects it with
  `.info()`, `.describe()`, and `.head()`, checks missing values, separates
  numerical vs. categorical columns, and closes with a short "data story"
  markdown summary.
- `titanic.csv` — the training dataset (891 passengers with ground-truth `Survived`
  labels) from the Kaggle competition
  [Titanic - Machine Learning from Disaster](https://kaggle.com/competitions/titanic).
- `test.csv` / `gender_submission.csv` — the accompanying test set (418 passengers,
  no ground truth) and Kaggle's example submission format, included for reference
  for a future modeling task.

### Citation
Will Cukierski. Titanic - Machine Learning from Disaster. https://kaggle.com/competitions/titanic, 2012. Kaggle.

### Toolkit
- Python 3
- Jupyter Notebook
- pandas
- NumPy

### How to run
```bash
pip install pandas numpy jupyter
jupyter notebook titanic_eda.ipynb
```
