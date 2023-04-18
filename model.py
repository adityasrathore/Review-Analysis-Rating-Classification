import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

train_df = pd.read_csv("cleaned_train.csv")
test_df = pd.read_csv("cleaned_train.csv")

print(train_df["Review"].isnull().sum())
print(test_df["Review"].isnull().sum())

# vectorize the text data using TfidfVectorizer with n-gram feature engineering
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # consider both unigrams and bigrams
X_train = vectorizer.fit_transform(train_df["Review"])
X_test = vectorizer.transform(test_df["Review"])

# build an Extra Trees classifier model on the training data
clf = ExtraTreesClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=50,  # reduce max_depth to address overfitting
    min_samples_leaf=10,  # increase min_samples_leaf to address overfitting
)

# Use BaggingClassifier with ExtraTreesClassifier as the base estimator
# to implement bagging ensemble method
clf = BaggingClassifier(
    base_estimator=clf,
    n_estimators=10,  # Number of base estimators in the ensemble
    random_state=42
)

# Use AdaBoostClassifier with ExtraTreesClassifier as the base estimator
# to implement boosting ensemble method
clf = AdaBoostClassifier(
    base_estimator=clf,
    n_estimators=50,  # Number of base estimators in the ensemble
    learning_rate=1.0,  # Learning rate for boosting
    random_state=42
)

clf.fit(X_train, train_df["Rating"])

train_pred = clf.predict(X_train)
train_acc = accuracy_score(train_df["Rating"], train_pred)
print(f"Accuracy on training data: {train_acc}")

# predict the ratings for the test data
test_pred = clf.predict(X_test)
# create a new dataframe with the predicted ratings and the review indexes
result_df = pd.DataFrame({"Id": range(len(test_pred)), "Rating": test_pred})

train_kappa = cohen_kappa_score(train_df["Rating"], train_pred)
print(f"Kappa score on training data: {train_kappa}")

# save the dataframe to a new CSV file
result_df.to_csv("sample_submission_emojis.csv", index=False)
