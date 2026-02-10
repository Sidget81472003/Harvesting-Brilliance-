import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle,warnings

print(" Loading and Preparing Data ")
df = pd.read_excel("Pumpkin_Seeds_Dataset.xlsx")

Q1 = df["Area"].quantile(0.25)
Q3 = df["Area"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df["Area"] >= lower_bound) & (df["Area"] <= upper_bound)]

columns_to_scale = ['Area', 'Perimeter', 'Major_Axis_Length']
scaler = MinMaxScaler()
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

df = df.drop(columns=['Convex_Area', 'Equiv_Diameter', 'Eccentricity', 'Minor_Axis_Length'])

le = LabelEncoder()
df['Class'] = le.fit_transform(df['Class'])

X = df.drop('Class', axis=1)
Y = df['Class']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=30)


#  Model 1: Logistic Regression 
print("\n Training Logistic Regression ")

logistic_regression = LogisticRegression()

logistic_regression.fit(x_train, y_train)

y_pred = logistic_regression.predict(x_test)

acc_lr = accuracy_score(y_test, y_pred)
c_lr = classification_report(y_test, y_pred)

print('Accuracy Score: ', acc_lr)
print(c_lr)


print("\n Random Forest ")
random_forest = RandomForestClassifier()
random_forest.fit(x_train, y_train)
y_pred_rf = random_forest.predict(x_test)

acc_rf = accuracy_score(y_test, y_pred_rf)
c_rf = classification_report(y_test, y_pred_rf)

print('Accuracy Score: ', acc_rf)
print(c_rf)

print("\n Decision Tree ")
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(x_train, y_train)
y_pred_dt = decision_tree_model.predict(x_test)
acc_dt = accuracy_score(y_test, y_pred_dt)
c_dt = classification_report(y_test, y_pred_dt)
print('Accuracy Score: ', acc_dt)
print(c_dt)

print("\n Multinomial Naive Bayes ")
NB = MultinomialNB()
NB.fit(x_train, y_train)
y_pred_nb = NB.predict(x_test)
acc_nb = accuracy_score(y_test, y_pred_nb)
c_nb = classification_report(y_test, y_pred_nb)
print('Accuracy Score: ', acc_nb)
print(c_nb)

print("\n Support Vector Machine (SVM) ")
support_vector = SVC()
support_vector.fit(x_train, y_train)
y_pred_svc = support_vector.predict(x_test)
acc_svc = accuracy_score(y_test, y_pred_svc)
c_svc = classification_report(y_test, y_pred_svc)
print('Accuracy Score: ', acc_svc)
print(c_svc)


print("\n Training Gradient Boosting ")
GBC = GradientBoostingClassifier()
GBC.fit(x_train, y_train)
y_pred_gbc = GBC.predict(x_test)

acc_gbc = accuracy_score(y_test, y_pred_gbc)
c_gbc = classification_report(y_test, y_pred_gbc)

print('Gradient Boosting Accuracy: ', acc_gbc)
print(c_gbc)

print("\n Manual Prediction Test ")
sample_input = [[0.410519, 0.340661, 0.294143, 0.9916, 0.7151, 0.8440, 1.7811, 0.7487]]

prediction = random_forest.predict(sample_input)

if prediction[0] == 0:
    print("Your seed lies in Çerçevelik class")
elif prediction[0] == 1:
    print("Your seed lies in Ürgüp Sivrisi class")


    print("\n Model Comparison ")
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 
              'Naive Bayes', 'Support Vector Machine', 'Gradient Boosting'],
    'Score': [acc_lr, acc_dt, acc_rf, acc_nb, acc_svc, acc_gbc]
})

# Sorting to find the best one
models = models.sort_values(by='Score', ascending=False)
print(models)



#dumping best model

pickle.dump(random_forest, open("model.pkl", "wb"))
print("✅ Model Saved Successfully!")