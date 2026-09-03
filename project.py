import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import dataframe_image as dfi # pip 3 to install in termina
from tensorflow import keras


train_data=pd.read_csv("composite_train.csv")
test_data = pd.read_csv("composite_test.csv")

pd.set_option('display.max_columns', None)

print("Training dataframe")
print(train_data.head())
print(train_data.info())

print("Testing dataframe")
print(test_data.head())
print(test_data.info())

print("Missing values checker for both csv files")
print(test_data.isnull().sum())
print(train_data.isnull().sum())

print("Duplicated rows:", train_data.duplicated().sum())
print("Duplicated rows:", test_data.duplicated().sum())

print(train_data.columns.tolist()) #finds the column name to use in the next step
print(test_data.columns.tolist())

#deletes any missing values
df_train=train_data.dropna()
df_test=test_data.dropna() 

 #converts text column into a string 
df_train['text']=df_train['text'].astype(str)
df_test['text']=df_test['text'].astype(str)

#seperates two rows 
X_train=df_train['text']
y_train=df_train['label']

X_test=df_test['text']
y_test=df_test['label']


#vectorize for the text columns to become numerical ones
vectorizer=TfidfVectorizer()
X_train=vectorizer.fit_transform(X_train) #learns the vocabulary and converts text into numbers
X_test=vectorizer.transform(X_test)

model=LogisticRegression()
model.fit(X_train,y_train)
pred_A = model.predict(X_test)

# shows the words which associate with vishing and non vishing predictions
print('start explainability')
feature_names = vectorizer.get_feature_names_out() #vectorizer  converts text into numerical features/gets word out ex('amazing..)
coef =model.coef_[0]#gets classifier coefficient/ the larger the value the stronger the influence 

coef_df = pd.DataFrame({"word": feature_names, 
                        "coefficient": coef
                        }) #creating a dataframe with words and their coef so positive or negative 

top_terms_vishing = (
    coef_df  # use ascending=false 
    .sort_values("coefficient",ascending=False) #use sort to give back the words with the strongest coefficient not just the rows 
    .head(15) #15 words with strongest coefficient
)

print("Words most associated with Vishing:")
print(top_terms_vishing)

top_non_vishing = (
    coef_df
    .sort_values("coefficient",ascending=True)
    .head(15)
)


print("Words most associated with Non-Vishing:")
print(top_non_vishing)


#shows the predicted and actual results and compares both 
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": pred_A
})
results["Correct"] = results["Actual"] == results["Predicted"]

final_table = results[["Actual", "Predicted", "Correct"]].head(30).style.hide(axis='index')

# 3. Save to image
dfi.export(final_table, 'selected_rows.png')

print("Accuracy:", accuracy_score(y_test, pred_A))
print(classification_report(y_test, pred_A))


#confusion matrix
def plot_cm(y_true, y_pred, title):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Non-Vishing", "Vishing"],
        cmap="Blues"
    )
    disp.ax_.set_title(title)
    plt.savefig(title)
    plt.show()

plot_cm(y_test, pred_A, "Logistic Regression-Confusion Matrix")




#model- MLP 
model_A= keras.models.Sequential([ 
keras.layers.Dense(50, activation="relu",input_shape=X_train.shape[1:]),
keras.layers.Dense(20, activation="relu"),
keras.layers.Dense(1,activation='sigmoid')
])

#adam optimizer used over SGD as it offers faster convergence
model_A.compile(loss="binary_crossentropy", 
                optimizer="adam",
                metrics=["accuracy"])

execution_history_A = model_A.fit(
    X_train,
    y_train,
    epochs=50, 
    validation_data=(X_test,y_test)
)

losses = pd.DataFrame(execution_history_A.history)
losses.plot()
plt.grid(True)
plt.show()

pred_MLP= model_A.predict(X_test) #for MLP different than regretion--gives back probability
binary_MLP = (pred_MLP > 0.5).astype(int).ravel() #converts probability to binary 

print("Accuracy:", accuracy_score(y_test,binary_MLP))
print(classification_report(y_test,binary_MLP))

plot_cm(y_test, binary_MLP, "Model A – Confusion Matrix")

plt.figure(figsize = (15,8))
plt.title("MLP Predicted VS actuall vihing ")
plt.plot(binary_MLP,'r-', label='Predicted')
plt.plot(y_test.values, 'g--', label='Actual')
plt.xlabel("Sample index")
plt.ylabel("Predicted vishing (0=non-vishing, Pass=vishing)")
plt.legend()
plt.savefig("Model_A.png")
plt.show()
