📞 Vishing Detection Using Machine Learning

A machine learning project that analyses text data to classify content as Vishing or Non-Vishing, using TF-IDF feature extraction, Logistic Regression, and a Multi-Layer Perceptron (MLP) neural network.


📌 Project Overview

This project investigates the use of machine learning for vishing detection.

The text data is transformed into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency). Two classification approaches are then trained and evaluated:

Logistic Regression
Multi-Layer Perceptron (MLP) neural network

The models are evaluated using accuracy, classification reports, and confusion matrices.

The project also includes a basic explainability analysis for the Logistic Regression model to identify words that have the strongest association with Vishing and Non-Vishing classifications.

🗂️ Repository Structure
vishing-detection/
│
├── data/
│   └── README.md
│
├── results/
│   ├── selected_rows.png
│   ├── Logistic Regression-Confusion Matrix.png
│   └── Model_A.png
│
├── vishing_detection.py
├── requirements.txt
└── README.md

Note: The training and testing CSV files are not stored in this repository because the dataset exceeds GitHub's file-size limit. The datasets are required locally to run the project.

🤖 Machine Learning Models
1. Logistic Regression

A Logistic Regression classifier is trained using the TF-IDF features.

The model is used to:

Classify text as Vishing or Non-Vishing
Measure classification accuracy
Generate a classification report
Produce a confusion matrix
Identify influential words through model coefficients
2. Multi-Layer Perceptron (MLP)

A neural network is also used for binary classification.

The architecture consists of:

Input Layer
     ↓
Dense Layer — 50 neurons — ReLU
     ↓
Dense Layer — 20 neurons — ReLU
     ↓
Output Layer — 1 neuron — Sigmoid

The model is trained using:

Binary Cross-Entropy loss
Adam optimiser
50 training epochs
Accuracy as an evaluation metric
🔤 Text Feature Extraction

The text data is converted into numerical features using TF-IDF.

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

The vectorizer learns the vocabulary from the training data and transforms the text into numerical feature vectors.

🔍 Model Explainability

The Logistic Regression model's coefficients are examined to identify words that have the strongest influence on the classification.

The project identifies:

Words most associated with Vishing

The words with the largest positive coefficients are extracted.

Words most associated with Non-Vishing

The words with the largest negative coefficients are extracted.

This provides an interpretable view of which terms the model considers important when making predictions.

📊 Model Evaluation

The models are evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix

Example evaluation:

print("Accuracy:", accuracy_score(y_test, pred_A))
print(classification_report(y_test, pred_A))
📈 Visualisations
Confusion Matrix

The confusion matrix shows the number of:

Correct Non-Vishing predictions
Correct Vishing predictions
Incorrect classifications
MLP Training History

The training history is plotted to compare model performance during training and validation.

Predicted vs Actual Results

The MLP predictions are compared with the actual test labels to investigate how closely the model's classifications match the test data.

🛠️ Technologies
Python — programming language
Pandas — data loading and manipulation
Scikit-learn — TF-IDF, Logistic Regression and evaluation
TensorFlow / Keras — MLP neural network
Matplotlib — data visualisation
Dataframe Image — exporting DataFrames as images
🚀 How to Run
1. Clone the repository
git clone YOUR_GITHUB_REPOSITORY_URL
cd vishing-detection
2. Install dependencies
pip install -r requirements.txt
3. Add the datasets

Place the required datasets on your local machine:

composite_train.csv
composite_test.csv

If your Python code expects the files in the current directory, place them in the same directory as the Python script.

4. Run the project
python vishing_detection.py
📁 Dataset

The project uses two datasets:

composite_train.csv — training data
composite_test.csv — testing data

The datasets are not included in this repository because the files exceed GitHub's recommended file-size limit.

💡 Future Improvements

Potential improvements to the project include:

Hyperparameter tuning for both models
Comparing additional classification algorithms
Improving the MLP architecture
Using cross-validation
Investigating class imbalance
Improving model explainability
Testing additional NLP techniques
Improving the visualisation of predicted versus actual classifications

👩‍💻 Author

Hind Michaal

Computer Science Student | Aspiring Software engineer / Machine Learning Practitioner/Interested in Cyber Security

⭐ Project Summary

This project demonstrates an end-to-end machine learning workflow for text classification:

Raw Text
   ↓
Data Cleaning
   ↓
TF-IDF Vectorisation
   ↓
Machine Learning Models
   ↓
Predictions
   ↓
Evaluation
   ↓
Visualisation & Explainability
