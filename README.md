# Netflix-Listings-Classification-Project

Project Description
The goal of this project was to build a machine learning model to classify Netflix listings as either "Movies" or "TV Shows" based on features such as release year, genre, rating, and duration. The dataset contained 6,234 entries with information about various shows on Netflix, including their type, genre, release year, rating, director, cast, and more.
I implemented a Logistic Regression model, a simple yet effective algorithm for binary classification tasks. The key steps involved in the project were:
1.	Data Preprocessing:
o	Handled missing values by replacing missing ratings with "Unknown."
o	One-hot encoded categorical variables to make them suitable for machine learning.
2.	Feature Engineering:
o	Selected relevant features: release_year, genre, rating, and duration.
o	Transformed categorical variables into numerical columns using one-hot encoding.
3.	Model Training and Evaluation:
o	Split the data into training (80%) and testing (20%) sets.
o	Trained a Logistic Regression model and evaluated it on the test set using accuracy, precision, recall, and F1-score.
o	Visualized model performance using a confusion matrix and analyzed feature importance.
________________________________________
Conclusion
The Logistic Regression model performed exceptionally well, achieving 100% accuracy on the test set. The classification report showed perfect precision, recall, and F1-scores for both classes (Movie and TV Show). The confusion matrix confirmed no misclassifications in the test data.
This high performance suggests that the features used (release year, genre, rating, and duration) provide strong separability between movies and TV shows in the dataset.
________________________________________
Areas for Improvement
While the model achieved perfect results, several areas can be explored to improve or validate the robustness of the solution:
1.	Check for Overfitting:
o	The perfect accuracy may indicate overfitting. Testing on additional unseen data or applying cross-validation can validate the model's robustness.
2.	Feature Reduction:
o	The one-hot encoding process resulted in 675 features. Feature selection or dimensionality reduction techniques (e.g., PCA) could simplify the model without sacrificing performance.
