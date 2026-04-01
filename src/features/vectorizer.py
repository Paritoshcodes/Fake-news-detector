from sklearn.feature_extraction.text import TfidfVectorizer

class Vectorizer:
    def __init__(self, max_features=10000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

    def fit(self, X_train):
        self.vectorizer.fit(X_train)

    def transform(self, X):
        return self.vectorizer.transform(X)

    def fit_transform(self, X_train):
        return self.vectorizer.fit_transform(X_train)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()