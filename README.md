📞 Vishing Detection Using Machine Learning

A machine learning research project investigating the detection of potential vishing scams and whether understandable explanations can be provided alongside detection results.

📌 Project Overview

Vishing, or voice phishing, is a form of social engineering where attackers attempt to obtain sensitive information by impersonating legitimate individuals or organisations.

This project investigates whether machine learning can be used to identify potential vishing scams from textual data.

The project also explores a potential research gap: instead of simply informing a user that a call is suspicious, could a system provide understandable reasons for why the call was flagged?

Potential characteristics include:

Suspicious language
Urgency or pressure
Requests for sensitive information
Impersonation
Other social-engineering indicators

🔬 Research Question

Can machine learning detect potential vishing scams while also providing understandable explanations of the characteristics that contributed to the classification?

Hypothesis

Machine learning can be used to detect potential vishing scams while identifying characteristics that contribute to the classification and presenting these characteristics in an understandable way.

🤖 Models Used

Logistic Regression

A Logistic Regression classifier is trained using TF-IDF text features.

Model coefficients are analysed to investigate which words or features are associated with different classifications.

Multi-Layer Perceptron (MLP)

A neural network is used as a second classification approach.

TF-IDF Features
       ↓
Dense Layer — 50 neurons
       ↓
Dense Layer — 20 neurons
       ↓
Sigmoid Output

🔤 Text Processing

The project uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert textual data into numerical features that can be used by the machine learning models.

📊 Evaluation

The models are evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion matrices

The project also investigates model explainability through analysis of influential features.

🔎 Research Areas

The research investigates:

What datasets are available for vishing detection?
What machine learning methods have been used previously?
What limitations exist in current vishing detection systems?
How are detection results presented to users?
Have existing systems incorporated explainability?
Could understandable explanations make detection results more useful to users?

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

The datasets are not included in this repository because they exceed GitHub's file-size limit.

🛠️ Technologies

Python
Pandas
Scikit-learn
TensorFlow / Keras
Matplotlib
Dataframe Image

🚀 Future Work
Investigate additional vishing datasets
Compare additional machine learning models
Improve NLP preprocessing
Investigate explainable AI techniques
Explore how explanations could be presented to users
Compare the project with existing vishing detection research

👩‍💻 Author

Hind Michaal
Computer Science Student/Machine Learning/Interested in Cyber Security
