from basichelpers import *
from sklearn.model_selection import cross_val_predict, StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, PassiveAggressiveClassifier, Perceptron, RidgeClassifier, RidgeClassifierCV, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

def select_classifier(X, y, n_splits=10, test_size=0.1, random_state=42, show=True):
    classifiers = [
        AdaBoostClassifier(),
        BaggingClassifier(),
        BernoulliNB(),
        CalibratedClassifierCV(),
        DecisionTreeClassifier(),
        ExtraTreeClassifier(),
        GaussianNB(),
        GaussianProcessClassifier(),
        GradientBoostingClassifier(),
        KNeighborsClassifier(),
        LinearDiscriminantAnalysis(),
        LinearSVC(),
        LogisticRegression(),
        LogisticRegressionCV(),
        MLPClassifier(),
        MultinomialNB(),
        NearestCentroid(),
        NuSVC(),
        PassiveAggressiveClassifier(),
        Perceptron(),
        QuadraticDiscriminantAnalysis(),
        RadiusNeighborsClassifier(),
        RandomForestClassifier(),
        RidgeClassifier(),
        RidgeClassifierCV(),
        SGDClassifier(),
        SVC()
    ]
    names = [clf.__class__.__name__ for clf in classifiers]
    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    scores = {}
    for i, (name, clf) in enumerate(zip(names, classifiers)):
        print('Processing {}...'.format(name))
        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            try:
                clf.fit(X_train, y_train)
                train_predictions = clf.predict(X_test)
                acc = accuracy_score(y_test, train_predictions)
            except:
                acc = 0
            s = scores.get(name, [])
            s.append(acc)
            scores[name] = s
    scores = [[n, np.mean(s)] for n, s in scores.items()]
    scores = pd.DataFrame(scores, columns=['Classifier', 'Score']).sort_values(by='Score', ascending=False)
    if show:
        print(scores)
    return scores.iloc[0, 0], classifiers[scores.iloc[0].name], scores