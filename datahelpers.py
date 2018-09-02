from basichelpers import *
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler, MaxAbsScaler, LabelEncoder, LabelBinarizer, OneHotEncoder

# Data cleaning:
#   Fix or remove outliers (optional).
#   Fill in missing values (e.g., with zero, mean, median…) or drop their rows (or columns).
# Feature selection (optional):
#   Drop the attributes that provide no useful information for the task.
# Feature engineering, where appropriate:
#   Discretize continuous features.
#   Decompose features (e.g., categorical, date/time, etc.).
#   Add promising transformations of features (e.g., log(x), sqrt(x), x^- [ ] etc.).
#   Aggregate features into promising new features.
# Feature scaling: standardize or normalize features.
##  0 mean and unit variance - scale, StandardScaler
##  0, 1 range - minmax_scale, MinMaxScaler 
##  -1, 1 - maxabs_scale, MaxAbsScaler (for data that is already centered at zero)
##  sparse data - maxabs_scale, MaxAbsScaler; scale, StandardScaler with "with_mean=False"
##  with outliers - robust_scale, RobustScaler


###############################################
## Data Preparation Classes

class DataFrameFiller(BaseEstimator, TransformerMixin):
    '''Docstring of `DataFrameFiller`.

    Deal with missing values for any 2D pandas DataFrame.

    Args:
        clean_mode: Determines how the missing data being processed.
            Avaiable values are None, "fill", "remove" and "both". Defaults to None.
            If None, default fill_mode will be applied to all columns:
            Integer - "median"; float - "mean"; string - "mode".
            If "remove", rows with missing data at all columns will be removed.
        fill_mode: Required when `clean_mode` set to "fill" or "both".
            Can be passed a single value or a dict.
            The dict must be structured as column_name -> mode.
            Valid modes for all column types are:
            Integer - "median", "mode", any integer or function that returns an integer;
            Float - "mean", "median", "mode", any float or function that returns a float;
            String - "mode", any string or function that returns a string;
            Missmatching modes are replaced by defaults.
            Functions must accept iterable and return values with types specified above.
            When passed a single mode, it will be applied to all columns.
            When passed a `dict`, the fill modes will be applied to the corresponding 
            columns. For the remain columns, if `clean_mode` is "fill", apply defaults,
            else if "remove", apply remove.
            When `case_mode` == "remove", this parameter will be ignored.
    '''
    
    def __init__(self, clean_mode: Optional[str] = None, fill_mode: Union[None, dict, str, int, float, Callable] = None):
        
        assert clean_mode in [None, 'fill', 'remove', 'both'], 'Invalid value for `clean_mode`. Avaliable values are None, "fill", "remove" and "both".'
        if clean_mode in ['both']:
            assert isinstance(fill_mode, dict), 'Invalid parameter. `fill_mode` must be a `dict` when `clean_mode` set to "both".'
        
        self.clean_mode = clean_mode
        self.fill_mode = clean_mode == 'remove' and None or fill_mode
        
    def fit(self, X: Union[pd.DataFrame, np.array], y=None):
        '''Initialize operators.
        
        patterns:
            1. clean_mode = None, use only fill_mode
            2. clean_mode = 'fill', fill_mode is single value, make and check filler dict with all columns;
            fill_mode is dict, check filler and add remain columns
            3. clean_mode = 'remove', ignore fill_mode
            4. clean_mode = 'both', fill_mode must be a dict, check filler and apply remove to remain columns
        
        Args:
            X: Accept a 2D pandas DataFrame, or a 1D Series or numpy array.
        '''
        if self.clean_mode == 'remove':
            self.fillers = 'remove'
        else:
            # make sure self.fill_mode is a dict 
            if not isinstance(self.fill_mode, dict):
                self.fill_mode = {col: self.fill_mode for col in X}
            # check for the validity of fill modes; default value will be used if not valid
            # then calculate the filler values
            fillers = {}
            for col, m in self.fill_mode.items():
                if m in ['mean', 'median', 'mode', None]:
                    valid_modes = self._valid_modes(X[col].dtype)
                    if not m in valid_modes: m = valid_modes[0]
                    m = m == 'mean' and X[col].mean() or (m == 'median' and X[col].median() or X[col].value_counts().index[0])
                if callable(m): m = m(X[col])
                fillers[col] = m
            if self.clean_mode == None:
                self.fillers = fillers
            elif self.clean_mode == 'fill':
                # default fillers for columns
                self.fillers = {col: X[col].dtype == np.dtype('O') and X[col].value_counts().index[0] or ( \
                                     np.issubdtype(X[col].dtype, np.integer) and X[col].median() or X[col].mean()) for col in X}
                self.fillers.update(fillers)
            else: # clean_mode = 'both'; apply remove to remain columns
                self.fillers = [fillers]
        return self
    def transform(self, X, y=None):
        X = X.copy()
        # # drop rows and columns where all values are NA
        # X = X.dropna(axis=0, how='all').dropna(axis=1, how='all')
        # clean missing value
        if self.fillers == 'remove':
            X = X.dropna()
        elif isinstance(self.fillers, list):
            X = X.fillna(value=self.fillers[0])
            X = X.dropna()
        else:
            X = X.fillna(value=self.fillers)
        return X
    
    def _valid_modes(self, dtype):
        return dtype == np.dtype('O') and ['mode',] \
            or (np.issubdtype(dtype, np.integer) and ['median', 'mode'] \
            or ['mean', 'median', 'mode'])

class DataFrameFeatureAdder(BaseEstimator, TransformerMixin):
    '''Docstring of `DataFrameFeatureAdder`

    Add extra features to DataFrame with given columns and functions.

    Args:
        adds: Tuples with same structure that contains any 
        number of columns' names, a list of new column names, 
        and a function to generate new columns with given 
        DataFrame and columns.
    '''

    def __init__(self, adds: list, remove: bool = False):
        self.adds = adds
        assert self.adds != [], 'Invalid input. At least one tuple for generating a new column is needed.'
        self.remove = remove

    def fit(self, X: pd.DataFrame, y=None):
        remove = self.remove and []
        for adds in self.adds:
            assert len(adds) == 3, 'Invalid parameters. Exactly 3 parameters are needed.'
            cols, names, func = adds
            for col in cols:
                assert col in X, '"{}" is not a column of given DataFrame'.format(col)
                if self.remove: remove.append(col)
            if isinstance(names, str): names = [names]
            for name in names:
                assert not name in X, '"{}" already exists.'.format(name)
        self.remove = remove
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X = X.copy()
        for adds in self.adds:
            cols, names, func = adds
            if isinstance(names, str): 
                X[names] = func(X, *cols)
            else:
                X = pd.concat([X, pd.DataFrame(func(X, *cols), columns=names)], axis=1)
            # X[names] = func(X, *cols)
        if self.remove:
            X = X.drop(columns=[*self.remove])
        return X

class DataFrameEncoder(BaseEstimator, TransformerMixin):
    '''Docstring of `DataFrameEncoder`

    Encodes specified columns with given methods.
    categorical -> numerical / one-hot
    numerical -> range / range one-hot / binarization

    Args:
        cols: Columns' names to encode.
        encoders: Encoders's names for encoding columns 
        corresponding to `cols`.
        newcolnames: A list of lists of string, that are
            encoded columns's names. Optional.
            If None, the combination of original column names 
            and unique values will be used.
            But if any one is specified, the others must be 
            specified with blank lists.
        params: A list of dict containing parameters for 
            each encoder. Optional.
            But if any one is specified, the others must be 
            specified with blank dicts.
        inplace: Whether delete the original columns or not
    '''

    def __init__(self, cols: list, encoders: list, newcolnames: Optional[List[List[str]]] = None, params: Optional[List[dict]] = None, inplace: bool = True):
        assert len(cols) == len(encoders), 'Parameter `cols` and `encoders` must have same length.'
        if newcolnames: assert len(newcolnames) == len(cols), 'Not enough new names. {} is needed.'.format(len(cols))
        else: newcolnames = [[]] * len(cols)
        if params: assert len(params) == len(cols), 'Not enough parameters. {} is needed.'.format(len(cols))
        else: params = [{}] * len(cols)
        self.cols = cols
        self.encoders = encoders
        self.newcolnames = newcolnames
        self.params = params
        self.inplace = inplace
    
    def fit(self, X: pd.DataFrame, y=None):
        encoders = {}
        for col, encoder, newnames, param in zip(self.cols, self.encoders, self.newcolnames, self.params):
            assert col in X, '"{}" is not a column of given DataFrame'.format(col)

            if encoder == 'label': # 1 col to 1 col
                encoder = LabelEncoder().fit(X[col])
                newnames = newnames or [col+'_labeled',]
            elif encoder == 'label2': # 1 col to multi cols
                encoder = LabelBinarizer(**{
                    k: param.get(k, d) for k,d in [
                        ('neg_label', 0),
                        ('pos_label', 1),
                        ('sparse_output', False)
                    ]
                }).fit(X[col])
                newnames = newnames or [col+'_'+ str(c) for c in encoder.classes_]
            elif encoder == '1hot': # 1 col to multi cols
                encoder = OneHotEncoder(**{
                    k: param.get(k, d) for k,d in [
                        ('n_values', 'auto'), ('categorical_features', 'all'), 
                        ('dtype', np.int), ('sparse', False), 
                        ('handle_unknown', 'ignore')]
                }).fit(X[col].values.reshape(-1,1))
                newnames = newnames or [col+'_'+ str(c) for c in sorted(X[col].unique())]
            elif callable(encoder):
                pass
            else:
                raise ValueError('"{}" is not a valid encoder. See help for more information.'.format(encoder))
            encoders[col] = (encoder, newnames)
        self.encoders = encoders
        return self
    
    def transform(self, X: pd.DataFrame, y=None):
        X = X.copy()
        for col in self.encoders:
            encoder, newnames = self.encoders[col]
            try:
                newcols = encoder.transform(X[col])
            except:
                newcols = encoder.transform(X[col].values.reshape(-1,1))
            X = X.reindex(columns=X.columns.tolist()+newnames)
            X[newnames] = newcols
        if self.inplace:
            X = X.drop(columns=list(self.encoders.keys()))
        return X

class DataFrameScaler(BaseEstimator, TransformerMixin):
    '''Docstring of `DataFrameScaler`

    Scaling a pandas DataFrame.

    Args:
        scale: Valid values are 
        "unit" - Centers to the mean and component wise scale 
        to unit variance;
        "0,1" - Scales data to given range. Use extra parameter 
        `feature_range=(min,max)` to set range;
        "-1,1" - Scales data to the range [-1, 1] by dividing 
        through the largest maximum value in each feature. 
        It is meant for data that is already centered at zero or 
        sparse data.
        Extra paramters will be passed to sklearn scalers if specified.
        ignore_cols: Columns that will not be scaled.
        By default, all categorical columns will be ignored.
        Specify this parameter to ignore numerical columns too.
    '''

    SCALERS = {
        'unit': StandardScaler,
        '0,1': MinMaxScaler,
        '-1,1': MaxAbsScaler
    }

    def __init__(self, scaler: str = 'unit', ignore_cols: List[str] = [], target_cols: List[str] = [], **kwargs):
        assert scaler in ['unit', '0,1', '-1,1'], 'Invalid scaler {}. See help for valid scalers.'.format(scaler)
        self.scaler = scaler
        self.ignore_cols = ignore_cols
        self.target_cols = ignore_cols and [] or target_cols
        self.kwargs = kwargs
        self.ignore_cols = ignore_cols
        self.target_cols = ignore_cols and [] or target_cols
    
    def fit(self, X: pd.DataFrame, y=None):
        if self.scaler == 'unit':
            self.scaler = StandardScaler(**{
                k: self.kwargs.get(k, d) for k,d in [('copy', True), ('with_mean', True), ('with_std', True)]
            })
        elif self.scaler == '0,1':
            self.scaler = MinMaxScaler(**{
                k: self.kwargs.get(k, d) for k, d in [('feature_range', (0, 1)), ('copy', True)]
            })
        elif self.scaler == '-1,1':
            self.scaler = MaxAbsScaler(**{
                k: self.kwargs.get(k, d) for k, d in [('copy', True)]
            })
        # self.scaler = self.SCALERS[self.scaler](**self.kwargs).fit(X)
        self.target_cols = self.ignore_cols and [col for col in X.select_dtypes(include=['number']) if not col in self.ignore_cols] or (
            self.target_cols and [col for col in X.select_dtypes(include=['number']) if col in self.target_cols] or X.select_dtypes(include=['number']).columns)
        self.scaler.fit(X[self.target_cols])
        return self
    
    def transform(self, X: pd.DataFrame, y=None):
        X = X.copy()
        X[self.target_cols] = self.scaler.transform(X[self.target_cols])
        return X