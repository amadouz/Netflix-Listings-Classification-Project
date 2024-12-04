
# coding: utf-8

# # Netflix Listings Classification Project
# 

# ### Project Description
# The goal of this project is to build a model to classify Netflix listings as either "Movies" or "TV Shows" based on features such as release year, genre, rating, and duration. The dataset contained 6,234 entries with information about various shows on Netflix, including their type, genre, release year, rating, director, cast, and more.
# I implemented a Logistic Regression model.
# 

# ### Import Librairies
# Import libraries for data manipulation, visualization, and modeling.

# In[26]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


# ### Load and Explore the data
# Load the Netflix dataset and explore its structure.

# In[27]:


df = pd.read_csv('netflix_data_extrat_credit_project.csv')


# Display Basic information about the data

# In[28]:


df.info()


# Display head and Tail of the dataset

# In[29]:


head = df.head()
tail = df.tail()
display(head, tail)


# ### Visualize the distribution of the target variable
# Plot the distribution of 'type' (Movie vs. TV Show)

# In[30]:


plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='type', palette='Set2')
plt.title('Distribution of Movies and TV Shows')
plt.xlabel('Type')
plt.ylabel('Count')
plt.show()


# ### Handle missing values
# Check for missing values

# In[31]:


print("Missing values before handling:")
print(df.isnull().sum())


# Fill missing values in the 'rating' column with 'Unknown'

# In[32]:


df['rating'] = df['rating'].fillna('Unknown')
display(df)


# Verify there are no missing values in relevant columns

# In[33]:


print("Missing values after handling:")
print(df.isnull().sum())


# ### Encode the target variable
# The 'type' column is the target variable (Movie or TV Show).
# Encode it as numeric: Movie = 1, TV Show = 0.

# In[34]:


label_encoder = LabelEncoder()
df['type_encoded'] = label_encoder.fit_transform(df['type'])


# # Visualize Release Year Trends
# Plot the distribution of release years

# In[35]:


plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='release_year', bins=20, kde=True, color='blue')
plt.title('Distribution of Release Years')
plt.xlabel('Release Year')
plt.ylabel('Count')
plt.show()


# ### Select Features and Encode Categorical Variables
# Features: 'release_year', 'genre', 'rating', 'duration'

# In[36]:


features = ['release_year', 'genre', 'rating', 'duration']


# One-hot encode categorical features ('genre', 'rating', 'duration')

# In[37]:


df_encoded = pd.get_dummies(df[features], columns=['genre', 'rating', 'duration'], drop_first=True)


# Define the feature set (X) and the target variable (y)

# In[38]:


X = df_encoded
y = df['type_encoded']


# Split the Data into Training and Testing Sets

# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### Train a Logistic Regression Model
# Initialize and train the Logistic Regression model

# In[40]:


logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train, y_train)


# Evaluate the Model (Make the predictions on the test set)

# In[41]:


y_pred = logistic_model.predict(X_test)


# Calculate Accuracy

# In[42]:


accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# Generate a classification report

# In[43]:


classification_report_summary = classification_report(y_test, y_pred)
print('\nClassification Report:\n', classification_report_summary)


# ### Visualize Confusion matrix
# Display the confusion 

# In[44]:


ConfusionMatrixDisplay.from_estimator(logistic_model, X_test, y_test, display_labels=['TV Show', 'Movie'], cmap='Blues')
plt.title('Confusion Matrix')
plt.show()


# ### Feature Importance
# Visualize the importance of top features

# In[45]:


coefficients = pd.Series(logistic_model.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
top_features = coefficients[:10]

plt.figure(figsize=(10, 6))
top_features.plot(kind='bar', color='orange')
plt.title('Top Features Impacting Classification')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.show()


# ### The Logistic Regression model achieves a good accuracy on the test set. The classification report showed perfect precision, recall, and F1-scores for both classes (Movie and TV Show). The confusion matrix confirmed no misclassifications in the test data.

# ### Areas of improvment : Several areas can be explored to improve or validate the robustness of the solution such as checking for Overfitting, feature reduction or handling potential data leakage

# ### FIN !!!!!
