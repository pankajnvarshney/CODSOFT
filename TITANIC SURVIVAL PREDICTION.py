#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Load the dataset
df = pd.read_csv("C:/Users/himan/Downloads/Titanic-Dataset.csv")


# In[4]:


df


# In[5]:


# Display basic information
print(df.info())


# In[6]:


# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())


# In[15]:


# Drop the age column
df.dropna(subset=["Age", "Embarked"], inplace=True)


# Reset index after dropping rows
df.reset_index(drop=True, inplace=True)


# In[16]:


# Verify missing values are removed
print("\nRemaining Missing Values:\n", df.isnull().sum())
print("\nUpdated Data Shape:", df.shape)


# In[17]:


# Set plot style
sns.set_style("whitegrid")


# In[18]:


# Countplot of survival
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df, palette="coolwarm")
plt.title("Survival Distribution")
plt.xlabel("Survived (1 = Yes, 0 = No)")
plt.ylabel("Count")
plt.show()


# In[19]:


# Survival by gender
plt.figure(figsize=(6,4))
sns.countplot(x="Sex", hue="Survived", data=df, palette="coolwarm")
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()


# In[20]:


# Survival by passenger class
plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", hue="Survived", data=df, palette="coolwarm")
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.show()


# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[26]:


# Drop unnecessary columns (PassengerId, Name, Ticket, Cabin)
df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True, errors='ignore')

# Convert categorical variables to numeric
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# Drop any remaining rows with missing values
df.dropna(inplace=True)

# Define features (X) and target variable (y)
X = df.drop(columns=["Survived"])
y = df["Survived"]


# In[27]:


# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[28]:


# Standardize the data (important for models like Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[30]:


# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", conf_matrix)


# ### Analysis of Model Performance
# 
# Overall Accuracy: 79.72%
# 
# The model correctly predicts survival ~80% of the time, which is a solid baseline for logistic regression.
# Precision vs. Recall:
# 
# For "Not Survived" (0):
# Precision (0.77): Out of all predicted non-survivors, 77% were actually correct.
# Recall (0.91): The model correctly identified 91% of the actual non-survivors.
# 
# For "Survived" (1):
# Precision (0.85): When predicting survival, the model is 85% correct.
# Recall (0.65): It only catches 65% of actual survivors, meaning some are misclassified as non-survivors.
# Confusion Matrix Interpretation:
# 
# True Negatives (73): Correctly classified non-survivors.
# 
# False Positives (7): Predicted as survivors but actually did not survive.
# 
# False Negatives (22): Predicted as non-survivors but actually survived.
# 
# True Positives (41): Correctly classified survivors.

# In[ ]:





# In[ ]:




