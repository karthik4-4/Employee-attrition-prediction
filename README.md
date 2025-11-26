# Employee Attrition Prediction

## 📋 Overview
This project is a machine learning-based application designed to predict the likelihood of employee attrition. By analyzing various workforce factors such as age, job role, income, and work history, the system helps HR departments identify employees at risk of leaving and take proactive retention measures.

## 🚀 Features
- **Interactive Web App**: User-friendly interface built with Streamlit for easy data input and prediction.
- **Real-time Prediction**: Instantly calculates attrition risk (High/Low) based on employee details.
- **Probability Score**: Provides the precise probability of an employee staying vs. leaving.
- **Comprehensive Analysis**: Includes a Jupyter Notebook documenting the data exploration, feature engineering, and model training process.

## 🛠️ Tech Stack
- **Language**: Python
- **Web Framework**: Streamlit
- **Machine Learning**: Scikit-learn (Bagging Classifier, Logistic Regression)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## 📂 Project Structure
- `app.py`: The main Streamlit application file.
- `Employee_attrition_prediction.ipynb`: Jupyter Notebook containing EDA and model development.
- `model.pkl`: Serialized trained machine learning model.
- `transformer.pkl`: Serialized data preprocessor/transformer.
- `report.html`: Automated data profiling report.

## 💻 How to Run
1. **Clone the repository** (if applicable) or download the files.
2. **Install dependencies**:
   Ensure you have Python installed, then install the required libraries:
   ```bash
   pip install streamlit pandas numpy scikit-learn matplotlib seaborn
   ```
3. **Run the application**:
   Navigate to the project directory and execute:
   ```bash
   streamlit run app.py
   ```
4. **Use the App**:
   Open the provided local URL in your browser, enter employee details, and click "Predict Attrition".

## 📊 Model Details
The system uses a **Bagging Classifier** with a **Logistic Regression** base estimator, optimized to handle class imbalance and provide robust predictions. The model was selected after evaluating multiple algorithms including Random Forest and Decision Trees.
