import pandas as pd
import numpy as np
from scipy import stats
from typing import Union, List, Callable, Optional, Tuple, Iterable
from sklearn.decomposition import PCA

## inference
def df_contingency_table(df: pd.DataFrame, col1: str, col2: str, ttype: str = 'count') -> pd.DataFrame:
    '''Docstring of `df_contingency_table`

    Construct a contigency table with two columns of given DataFrame.

    Args:
        df: A pandas DataFrame.
        col1: Index of the contigency table.
        col2: Column of the contigency table.
        ttype: Determines how the contigency table is calculated.
            'count': The counts of every combination.
            'colper': The percentage of every combination to the 
            sum of every rows.
            Defaults to 'count'.
    
    Returns:
        crosstab: DataFrame.
    '''
    ct = pd.crosstab(df[col1], df[col2])
    # add sum column and row
    ct['Total'] = ct.sum(axis=1)
    ct.loc['Total'] = ct.sum()
    if ttype == 'count': pass
    elif ttype == 'colper': # row percentage
        ct = (ct / ct.loc['Total'] * 100).round().astype(int)
    return ct

def df_chi_square(df: pd.DataFrame, col1: str, col2: str, cramersv: bool = True) -> str:
    '''Docstring of `df_chi_square`

    Run chi-square test on two columns of given pandas DataFrame,
    with a contigency table calculated on the two columns.

    Args:
        df: A pandas DataFrame.
        col1: Index of the contigency table.
        col2: Column of the contigency table.
        cramersv: Whether to calculate the Cramér’s V.
            Defaults to True.
    
    Returns:
        A string describes the result of the chi-square test.
    '''
    cto = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, exp = stats.chi2_contingency(cto)
    if cramersv:
        cramersv = '<br>Cramér’s V = {:.2f}'.format(corrected_cramers_V(chi2, cto))
    return 'χ2({dof}) = {chi2:.2f}<br>p = {p:.3f}{cramersv}'.format(dof=dof, chi2=chi2, p=p, cramersv=cramersv or '' )

def df_chi_square_matrix(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    '''Docstring of `df_chi_square_matrix`

    Run chi-square test on every two columns of given pandas DataFrame,
    with contigency tables calculated on the two columns.

    Args:
        df: A pandas DataFrame.
        columns: A list contains columns to run chi-square test with.
    
    Returns:
        A pandas DataFrame of strings descibe the results of chi-square test.
    '''
    dfr = pd.DataFrame(index=columns, columns=columns)
    for i in range(len(columns)):
        for j in range(len(columns)):
            dfr.iloc[i, j] = df_chi_square(df, columns[i], columns[j])
    return dfr

def cramers_V(chi2: float, ct: pd.DataFrame) -> float:
    '''Docstring of `cramers_V`.

    Original algorithm of Cramer's V.

    Args:
        chi2: The chi-square statistic of `ct`.
        ct: The contigency table to calculate Cramer's V for .
    
    Returns:
        The Cramer's V statistic.
    '''
    r, k = ct.shape
    return np.sqrt(chi2 / (ct.values.sum() * (min(r, k) - 1)))

def corrected_cramers_V(chi2: float, ct: pd.DataFrame) -> float:
    '''Docstring of `corrected_cramers_V`.

    Corrected algorithm of Cramer's V.
    See https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V for more details.

    Args:
        chi2: The chi-square statistic of `ct`.
        ct: The contigency table to calculate Cramer's V for .
    
    Returns:
        The Cramer's V statistic.
    '''
    n = ct.values.sum()
    phi2 = chi2/n
    r, k = ct.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

## data group
def grouped_col(df: pd.DataFrame, col1: str, col2: str) -> Tuple[np.array, List[pd.Series]]:
    '''Docstring of `grouped_col`

    Group one column based on the unique values of another.

    Args:
        df: A pandas DataFrame.
        col1: Name of the column that the unique values of which
            will be used for grouping.
        col2: Name of the column that will be grouped.
    
    Returns:
        A (2, ) tuple contains a numpy array and a list.
        The array contains the unique values.
        The list contains grouped pandas Series.
    '''
    vals = df[col1].unique()
    return vals, [df[df[col1]==val][col2].dropna() for val in vals]

## descriptive
def advanced_describe(data: list, index=None, dtype=None, name: str = None) -> pd.Series:
    '''Docstring of `advanced_describe`

    Descriptive statistics of given data.
    Those are "count", "mean", "std", "min", "25%-50%-75% quantile", "max", "variance", "skewness", "kurtosis".

    Args:
        data: A pandas Series or a list or a numpy array of numerical data.
        index: array-like or Index (1d)
        dtype: numpy.dtype or None
        name: The name of the data.

    Returns:
        The descriptive statistics of given data
    '''
    data = pd.Series(data, index=index, dtype=dtype, name=name)
    des = data.describe()
    t = stats.describe(data)
    des['variance'] = t.variance
    des['skewness'] = t.skewness
    des['kurtosis'] = t.kurtosis
    des = np.r_[[['', name]], [['Item', 'Statistic']], des.reset_index().values]
    return des

## linear algebra
def explained_variance(X: Optional[np.array] = None, raito: bool = False, cumulative: bool = True) -> Union[np.array, List[np.array]]:
    '''Docstring of `explained_variance`

    Calculate the explained variances of given data.

    Args:
        X: A numpy array for calculation.
        ratio: Whether to return the values as percentage 
        to the sum or not.
        cumulative: If True, return the cumulative sum of
        explained variances altogether.

    Returns:
        Explained variances or Explained variances and its
        cumulative sum.
    '''
    pca = PCA()
    pca.fit(X)
    var_exp = ratio and pca.explained_variance_ratio_ or pca.explained_variance_
    if cumulative:
        cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance
        return var_exp, cum_var_exp
    return var_exp
