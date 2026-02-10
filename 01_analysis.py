import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from  sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

print(" Reading Dataset ")
try:
    df = pd.read_excel("Pumpkin_Seeds_Dataset.xlsx")
    print(" Dataset Loaded Successfully!")
except FileNotFoundError:
    print("File not found. Ensure 'Pumpkin_Seeds_Dataset.xlsx' is in the folder.")
    exit()

print("\n Data Info ")
df.info()

print("\n Data Shape (Rows, Columns) ")
print(df.shape)

print("\n Missing Values Count ")
print(df.isnull().sum())


#saperate

print("\n Visualizing Outliers (Area) ")
sns.boxplot(x=df['Area'])
plt.title("Before Outlier Removal")
plt.show()

print("\n Removing Outliers ")
Q1 = df["Area"].quantile(0.25)
Q3 = df["Area"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df["Area"] >= lower_bound) & (df["Area"] <= upper_bound)]

print("Outliers Removed")

print("\n Verifying Removal ")
sns.boxplot(x=df['Area'])
plt.title("After Outlier Removal")
plt.show()

print("\n New Data Shape ")
print(df.shape)

print("\n Scaling Features ")
columns_to_scale = ['Area', 'Perimeter', 'Major_Axis_Length']

scaler = MinMaxScaler()
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

print("Scaling Complete. First 5 rows:")
print(df.head())


print("\n Dropping Columns ")
df = df.drop(columns=['Convex_Area', 'Equiv_Diameter', 'Eccentricity', 'Minor_Axis_Length'])

print("Columns Dropped. New Dataframe Head:")
print(df.head())

##all analysis

##1 Descriptive Analysiss
print(df.describe())

##2Univariate Analysis
print("\n Univariate Analysis: Class Count ")
plt.figure(figsize=(6, 6))
sns.countplot(data=df, x='Class')
plt.title("Distribution of Pumpkin Seed Varieties")
plt.show()

print(" Analysis Complete.")

print("\n Modified Bivariate Analysis: Area vs Perimeter (by Class) ")
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Area', y='Perimeter', hue='Class')
plt.title('Scatter Plot: Area vs Perimeter (Separated by Variety)')
plt.grid(True)
plt.show()

print("Analysis Complete.")

le = LabelEncoder()
df['Class'] = le.fit_transform(df['Class'])

print("\n Multivariate Analysis: Heatmap ")
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, linewidths=0.2)
plt.title("Correlation Matrix")
plt.show()

print("\n Splitting Data into X and Y ")
X = df.drop('Class', axis=1)
Y = df['Class']

print("X Head:")
print(X.head())
print("\nY Head:")
print(Y.head())

print("\n Splitting into Train and Test ")
# Using random_state=30 as shown in your screenshot
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=30)

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)