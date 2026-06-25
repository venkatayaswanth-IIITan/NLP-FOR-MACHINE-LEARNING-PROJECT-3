# Kindle Review Sentiment Analysis Project – Code Explanation

This project performs **Sentiment Analysis** on Kindle book reviews using:

1. Data Preprocessing
2. Text Cleaning
3. Lemmatization
4. Train-Test Split
5. Feature Extraction (BoW & TF-IDF)
6. Naive Bayes Classification
7. Model Evaluation

Dataset details are from the Kindle Reviews dataset. 

---

# Step 1: Import Dataset

```python
import pandas as pd
data = pd.read_csv('Kindle Reviews/all_kindle_review.csv')
data.head()
```

### What it does?

* Loads the CSV file into a DataFrame.
* Displays first 5 rows.

### Sample Columns

| Column       | Meaning            |
| ------------ | ------------------ |
| reviewText   | User review        |
| rating       | Rating (1-5 stars) |
| reviewerName | Reviewer name      |
| summary      | Review summary     |

---

# Step 2: Select Required Columns

```python
df = data[['reviewText','rating']]
df.head()
```

### Why?

For sentiment analysis we only need:

* Review Text → Input (X)
* Rating → Target (Y)

---

# Step 3: Check Dataset Shape

```python
df.shape
```

Output:

```python
(12000, 2)
```

Meaning:

* 12000 reviews
* 2 columns

---

# Step 4: Check Missing Values

```python
df.isnull().sum()
```

Output:

```python
reviewText    0
rating        0
```

Meaning:

No missing values.

---

# Step 5: Check Unique Ratings

```python
df['rating'].unique()
```

Output:

```python
[3,5,4,2,1]
```

Ratings range from 1 to 5 stars.

---

# Step 6: Count Ratings

```python
df['rating'].value_counts()
```

Output:

```python
5    3000
4    3000
3    2000
2    2000
1    2000
```

### Meaning

Dataset contains:

* 3000 positive reviews (5-star)
* 3000 positive reviews (4-star)
* 2000 neutral reviews (3-star)
* 2000 negative reviews (2-star)
* 2000 negative reviews (1-star)

---

# Step 7: Convert to Binary Classification

```python
df['rating']=df['rating'].apply(
    lambda x:0 if x<3 else 1
)
```

### Logic

| Rating | Sentiment    |
| ------ | ------------ |
| 1      | Negative (0) |
| 2      | Negative (0) |
| 3      | Positive (1) |
| 4      | Positive (1) |
| 5      | Positive (1) |

### Example

```python
1 → 0
2 → 0
3 → 1
4 → 1
5 → 1
```

---

# Step 8: Check New Distribution

```python
df['rating'].value_counts()
```

Output:

```python
1    8000
0    4000
```

Meaning:

* Positive Reviews = 8000
* Negative Reviews = 4000

---

# Step 9: Convert Text to Lowercase

```python
df['reviewText']=df['reviewText'].str.lower()
```

### Example

Before:

```text
This Book Is Amazing
```

After:

```text
this book is amazing
```

### Why?

Avoids treating:

```text
Book
BOOK
book
```

as different words.

---

# Step 10: Import NLP Libraries

```python
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
```

### Purpose

* `re` → Regular Expressions
* `nltk` → NLP library
* `stopwords` → Common useless words

Examples:

```text
is
am
the
a
an
are
```

---

# Step 11: Import BeautifulSoup

```python
from bs4 import BeautifulSoup
```

Used for removing HTML tags.

Example:

```html
<p>Hello</p>
```

becomes

```text
Hello
```

---

# Step 12: Remove Special Characters

```python
df['reviewText']=df['reviewText'].apply(
    lambda x: re.sub('[^a-z A-z 0-9-]+','',x)
)
```

### Example

Before:

```text
Amazing!!! Book@@@
```

After:

```text
Amazing Book
```

---

# Step 13: Remove Stopwords

```python
df['reviewText']=df['reviewText'].apply(
 lambda x:" ".join(
 [y for y in x.split()
  if y not in stopwords.words('english')]
 )
)
```

### Example

Before

```text
this is a very good book
```

After

```text
good book
```

---

# Step 14: Remove URLs

```python
df['reviewText']=df['reviewText'].apply(
 lambda x: re.sub(url_pattern,'',str(x))
)
```

### Example

Before

```text
visit https://amazon.com
```

After

```text
visit
```

---

# Step 15: Remove HTML Tags

```python
df['reviewText']=df['reviewText'].apply(
 lambda x: BeautifulSoup(x,'lxml').get_text()
)
```

### Example

Before

```html
<b>Good Book</b>
```

After

```text
Good Book
```

---

# Step 16: Remove Extra Spaces

```python
df['reviewText']=df['reviewText'].apply(
 lambda x:" ".join(x.split())
)
```

### Example

Before

```text
good      book
```

After

```text
good book
```

---

# Step 17: Lemmatization

## Import

```python
from nltk.stem import WordNetLemmatizer
```

## Create Object

```python
lemmatizer=WordNetLemmatizer()
```

## Function

```python
def lemmatize_words(text):
    return " ".join(
      [lemmatizer.lemmatize(word)
       for word in text.split()]
    )
```

## Apply

```python
df['reviewText']=df['reviewText'].apply(
 lambda x: lemmatize_words(x)
)
```

### Example

Before

```text
books
cars
running
```

After

```text
book
car
running
```

### Why?

Reduces vocabulary size.

---

# Step 18: Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=
train_test_split(
df['reviewText'],
df['rating'],
test_size=0.20
)
```

### Meaning

80% → Training

```text
9600 reviews
```

20% → Testing

```text
2400 reviews
```

---

# Step 19: Bag of Words (BoW)

```python
from sklearn.feature_extraction.text import CountVectorizer

bow = CountVectorizer()

X_train_bow = bow.fit_transform(X_train).toarray()

X_test_bow = bow.transform(X_test).toarray()
```

---

## How BoW Works?

Reviews:

```text
good book
bad book
```

Vocabulary:

```text
good
bad
book
```

Vectors:

| Review    | good | bad | book |
| --------- | ---- | --- | ---- |
| good book | 1    | 0   | 1    |
| bad book  | 0    | 1   | 1    |

Machine learning algorithms need numbers, so text becomes vectors.

---

# Step 20: TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer()

X_train_tfidf=tfidf.fit_transform(X_train).toarray()

X_test_tfidf=tfidf.transform(X_test).toarray()
```

---

## Difference from BoW

BoW:

```text
Counts words
```

TF-IDF:

```text
Importance of words
```

Common words get lower weight.

Important words get higher weight.

---

# Step 21: Train Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB

nb_model_bow=
GaussianNB().fit(
X_train_bow,
y_train
)

nb_model_tfidf=
GaussianNB().fit(
X_train_tfidf,
y_train
)
```

### What is Naive Bayes?

Probability-based classifier.

Uses:

```text
P(Positive | Review)
P(Negative | Review)
```

Predicts class with highest probability.

---

# Step 22: Prediction

### BoW

```python
y_pred_bow=
nb_model_bow.predict(
X_test_bow
)
```

### TF-IDF

```python
y_pred_tfidf=
nb_model_tfidf.predict(
X_test_tfidf
)
```

⚠️ Your notebook has a mistake:

```python
y_pred_tfidf=nb_model_bow.predict(X_test_tfidf)
```

Should be:

```python
y_pred_tfidf=nb_model_tfidf.predict(X_test_tfidf)
```

---

# Step 23: Evaluate Model

```python
from sklearn.metrics import (
confusion_matrix,
accuracy_score,
classification_report
)
```

---

# BoW Results

```python
accuracy_score(
y_test,
y_pred_bow
)
```

Output:

```python
58.33%
```

---

## Confusion Matrix

```python
[[511 308]
 [692 889]]
```

Meaning:

| Actual           | Predicted |
| ---------------- | --------- |
| Negative Correct | 511       |
| Negative Wrong   | 308       |
| Positive Wrong   | 692       |
| Positive Correct | 889       |

---

# TF-IDF Results

```python
58.16%
```

Confusion Matrix:

```python
[[502 317]
 [687 894]]
```

---

# Complete Project Flow

```text
Kindle Reviews Dataset
          ↓
Select Review & Rating
          ↓
Convert Ratings → Binary
          ↓
Lowercase
          ↓
Remove Special Characters
          ↓
Remove Stopwords
          ↓
Remove URLs
          ↓
Remove HTML
          ↓
Remove Extra Spaces
          ↓
Lemmatization
          ↓
Train-Test Split
          ↓
BoW / TF-IDF
          ↓
Naive Bayes
          ↓
Prediction
          ↓
Accuracy & Confusion Matrix
```

# Interview Question

**Why do we use BoW and TF-IDF before ML algorithms?**

Machine Learning algorithms cannot understand text directly.

Example:

```text
"This book is amazing"
```

must become:

```text
[0.34, 0.21, 0.89, ...]
```

BoW and TF-IDF convert text into numerical vectors so algorithms like Naive Bayes, Logistic Regression, SVM, Random Forest, etc., can learn patterns from the reviews.
