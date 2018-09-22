from basichelpers import *
from sklearn.model_selection import \
    cross_val_predict, StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV, \
    cross_val_score, train_test_split, learning_curve


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
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.DataFrame):
        y = y.values
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


from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.linear_model import HuberRegressor, PassiveAggressiveRegressor, RANSACRegressor, SGDRegressor, TheilSenRegressor, ARDRegression, LinearRegression, LogisticRegression, Lasso, Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cross_decomposition import PLSRegression


def select_regressor(X, y, scoring='neg_mean_squared_error', show=True):
    regressors = [
        AdaBoostRegressor(),
        # ARDRegression(),
        BaggingRegressor(),
        DecisionTreeRegressor(),
        ElasticNet(),
        ExtraTreeRegressor(),
        ExtraTreesRegressor(),
        # GaussianProcessRegressor(),
        GradientBoostingRegressor(),
        HuberRegressor(),
        KNeighborsRegressor(),
        Lasso(),
        LinearRegression(),
        # LogisticRegression(),
        MLPRegressor(),
        PassiveAggressiveRegressor(),
        PLSRegression(),
        # RadiusNeighborsRegressor(),
        RandomForestRegressor(),
        RANSACRegressor(),
        Ridge(),
        SGDRegressor(),
        TheilSenRegressor(),
    ]
    names = [reg.__class__.__name__ for reg in regressors]
    # cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    scores = {}
    for i, (name, reg) in enumerate(zip(names, regressors)):
        print('Processing {}...'.format(name))
        ss = cross_val_score(reg, X, y, scoring=scoring, cv=10)
        scores[name] = ss
        # for train_index, test_index in cv.split(X, y):
        #     X_train, X_test = X[train_index], X[test_index]
        #     y_train, y_test = y[train_index], y[test_index]
        #     try:
        #         clf.fit(X_train, y_train)
        #         train_predictions = clf.predict(X_test)
        #         rmse = np.sqrt(mean_squared_error(y_test, train_predictions))
        #     except:
        #         rmse = 0
        #     s = scores.get(name, [])
        #     s.append(acc)
        #     scores[name] = s
    scores = [[n, np.sqrt(-s).mean()] for n, s in scores.items()]
    scores = pd.DataFrame(scores, columns=['Regressor', 'Score']).sort_values(by='Score', ascending=True)
    if show:
        print(scores)
    return scores.iloc[0, 0], regressors[scores.iloc[0].name], scores



def simple_model_scores(model, X_train, y_train, X_test=None, y_test=None, regression=True, **kwargs):
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.values
    print('Model:')
    print(' ', model.__class__.__name__)
    if regression:
        scores = np.sqrt(-cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=10))
    else:
        scores = []
        cv = StratifiedShuffleSplit(n_splits=kwargs.get('n_splits', 10), test_size=kwargs.get('test_size', 0.1), random_state=kwargs.get('random_state', 42))
        for train_index, test_index in cv.split(X_train, y_train):
            X_t, X_v = X_train[train_index], X_train[test_index]
            y_t, y_v = y_train[train_index], y_train[test_index]
            try:
                model.fit(X_t, y_t)
                train_predictions = model.predict(X_v)
                acc = accuracy_score(y_v, train_predictions)
            except:
                acc = 0
            scores.append(acc)
    
    print('Cross-Valition score:')
    print(' mean:', np.mean(scores), 'std:', np.std(scores))

    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    print('Train set score:')
    if regression:
        rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        print(' ', rmse)
    else:
        acc = accuracy_score(y_train, y_train_pred)
        print(' ', acc)

    if not X_test is None:
        y_pred = model.predict(X_test)
        print('Test data score:')
        if regression:
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(' ', test_rmse)
        else:
            test_acc = accuracy_score(y_test, y_pred)
            print(' ', test_acc)

def simple_grid_search_scores(model, params, X_train, y_train, cv=10, scoring='neg_mean_squared_error', verbose=1):
    grid = GridSearchCV(model, params, cv=cv, scoring=scoring, verbose=verbose)
    grid.fit(X_train, y_train)
    print('== Grid Search ================')
    print('Best parameters:')
    print(' ', grid.best_params_)
    return grid

def learning_curve_gen(
        estimator, 
        X, 
        y, 
        cv=None,
        train_sizes=np.linspace(.1, 1.0, 5), 
        **kwargs):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, train_sizes=train_sizes, n_jobs=kwargs.get('n_jobs', 1))
    train_means = np.mean(train_scores, axis=1)
    train_stds = np.std(train_scores, axis=1)
    val_means = np.mean(test_scores, axis=1)
    val_stds = np.std(test_scores, axis=1)
    return train_sizes, train_means, train_stds, val_means, val_stds