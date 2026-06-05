# ML Project 02 – Adult Income Classification

## Objective

This project predicts whether a person earns more than `$50K` per year using the Adult Census Income dataset.

This is a binary classification problem.

Target variable:

- `<=50K` → 0
- `>50K` → 1

The project follows a similar workflow to the Customer Churn neural network exercise.

---

## Project Structure

```text
Adultincomeclassification/
│
├── data/
│   └── adult.csv
│
├── models/
│
├── notebook/
│   └── 01_adult_income_classification.ipynb
│
├── reports/
│   ├── model_comparison.csv
│   ├── model_comparison.txt
│   ├── notebook_model_comparison.csv
│   ├── model_comparison_bar_chart.png
│   ├── tensorflow_confusion_matrix.png
│   └── pytorch_confusion_matrix.png
│
├── src/
│   ├── 01_data_check.py
│   └── 02_full_model_workflow.py
│
├── .gitignore
└── README.md