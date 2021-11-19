import pandas as pd
import numpy as np
df = pd.read_csv("C:/Users/harsh/Downloads/SpamClassifier-master/SpamClassifier-master/smsspamcollection"
                 "/SMSSpamCollection", sep='\t', names=["Label", "Message"])
print(df.head())
import re
from nltk.stem import WordNetLemmatizer
lemmatise = WordNetLemmatizer()
from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# ps = PorterStemmer()
text = []
for i in range(len(df)):
    line = ''
    message = re.sub('[^a-zA-Z]', string=df['Message'][i], repl=' ')
    message = message.lower()
    message = message.split()
    for word in message:
        if word not in stopwords.words("english"):
            # word = ps.stem(word)
            word = lemmatise.lemmatize(word)
            line += (word + " ")
    text.append(line)
# print(len(text), text[0])
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=3000)
X = cv.fit_transform(text).toarray()
print(X.shape)
y = df['Label']
y = np.array(y.tolist())
# print(type(X), type(y), y[0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print(X_test.shape, X_train[0])
# print(y_train.shape, y_test.shape)
from sklearn.naive_bayes import MultinomialNB
Spam_check_model = MultinomialNB().fit(X_train, y_train)
pred = Spam_check_model.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
print("Confusion Matrix: ", confusion_matrix(y_test, pred))
print("Accuracy: ", accuracy_score(y_test, pred))
