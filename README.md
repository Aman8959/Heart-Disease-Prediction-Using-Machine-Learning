# Heart Disease Prediction Using Machine Learning

## Overview
This project implements a machine learning classification model to predict the presence of heart disease in patients using medical datasets and data science techniques. The model leverages various classification algorithms to achieve accurate predictions based on patient health indicators.

## Project Objective
- Build a robust predictive model to identify patients at risk of heart disease
- Analyze medical features that influence heart disease diagnosis
- Implement and compare multiple machine learning algorithms
- Provide accurate predictions using patient medical data

## Dataset
The project uses medical datasets containing patient health metrics such as:
- Age
- Gender
- Cholesterol levels
- Blood pressure
- Maximum heart rate achieved
- ST depression
- And other relevant cardiac indicators

## Technologies & Libraries Used
- **Python 3.x** - Programming language
- **Jupyter Notebook** - Development environment
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib/Seaborn** - Data visualization

## Machine Learning Approaches
The project implements and evaluates various classification algorithms:
- Logistic Regression
- Decision Trees
- Random Forest
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting

## Project Structure
```
├── README.md
├── notebooks/
│   └── Heart_Disease_Prediction.ipynb
├── data/
│   └── heart_disease_dataset.csv
└── requirements.txt
```

## Key Features
- ✅ Data preprocessing and cleaning
- ✅ Exploratory Data Analysis (EDA)
- ✅ Feature scaling and normalization
- ✅ Model training and evaluation
- ✅ Cross-validation for model robustness
- ✅ Performance metrics (Accuracy, Precision, Recall, F1-Score)
- ✅ Comparison of multiple algorithms

## Getting Started

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook

### Installation
1. Clone the repository:
```bash
git clone https://github.com/Aman8959/Heart-Disease-Prediction-Using-Machine-Learning.git
cd Heart-Disease-Prediction-Using-Machine-Learning
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Open and run `Heart_Disease_Prediction.ipynb`

## Results
The trained models achieve competitive performance metrics with the best-performing algorithm providing:
- High accuracy in predicting heart disease presence
- Balanced precision and recall
- Robust generalization to unseen data

## Usage
Simply run the Jupyter notebook cells sequentially to:
1. Load and explore the dataset
2. Preprocess the data
3. Train multiple models
4. Evaluate and compare results
5. Visualize predictions and model performance

## Future Enhancements
- Deploy the model as a web API
- Implement deep learning approaches
- Add cross-validation with different train-test splits
- Integrate real-time prediction capability
- Create a web-based user interface

## Contributing
Contributions are welcome! Please feel free to submit issues and enhancement requests.

## License
This project is open source and available under the MIT License.

## Author
**Aman8959**

## Acknowledgments
- Dataset sourced from publicly available medical databases
- Inspired by machine learning best practices in healthcare analytics

---

**Note:** This project is for educational purposes. For medical diagnosis, always consult qualified healthcare professionals.
