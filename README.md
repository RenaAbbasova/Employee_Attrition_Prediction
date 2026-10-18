# Employee attrition: exploratory analysis and a PyTorch neural network

Two notebooks on an HR employee attrition dataset (about 16% of the employees leave the company).

| Notebook | What it does |
|---|---|
| `EDA_HR_employee_attrition.ipynb` | Exploratory analysis: attrition rate, departments, job levels, education, work-life balance. |
| `HR_ employee attrition.ipynb` | Preprocessing (outlier capping with the IQR rule, scaling, encoding) and a sequential neural network in PyTorch. |

## Status

A learning project from the Learnbay program. These are limitations I know about and plan to fix:

- The numeric columns are **scaled before the train/test split**, so information from the test set leaks into the scaler.
- The model is judged mostly by **accuracy**, although only about 16% of employees leave. A model that always answers "stays"
  would already be about 84% accurate.

The next version will split first, add a baseline and report recall, precision and PR-AUC, as in my
[Telecom churn project](https://github.com/RenaAbbasova/Telecom-project).

## Data

`HR_Employee_Attrition_Data.csv`, an HR employee attrition dataset. It is not stored in this repository.
