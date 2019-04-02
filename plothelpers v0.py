from basichelpers import *
import colorlover as cl
import plotly.offline as plty
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from graphviz import Source
from sklearn.tree import export_graphviz
from sklearn.base import clone

## colorlover colors:
## https://plot.ly/ipython-notebooks/color-scales/
## plotly colors:
## 'Blackbody', 'Bluered', 'Blues', 'Earth', 'Electric', 'Greens', 'Greys','Hot',
## 'Jet', 'Picnic', 'Portland', 'Rainbow', 'RdBu', 'Reds', 'Viridis', 'YlGnBu', 'YlOrRd'

###############################################################################
## Constants
###############################################################################
line_styles = ['solid', 'longdashdot', 'longdash', 'dashdot', 'dash', 'dot']
###############################################################################
## Helper functions
###############################################################################
def make_colorscale(
        clname:     str = '3,qual,Paired', 
        n:          Optional[int] = None, 
        reverse:    bool = False) -> list:
    cnum, ctype, cname = clname.split(',')
    colors = cl.scales[cnum][ctype][cname]
    if n and n > int(cnum): 
        colors = cl.to_rgb(cl.interp(colors, (n // 10 + 1) * 10))
    else:
        n = len(colors)
    if reverse: colors = colors[::-1]
    n_colors = len(colors)
    step = n_colors // n
    return [[i/(n-1), c] for i, c in  enumerate(colors) if (i+1) % step == 0]

###############################################################################
## Plotly DataFrame plot functions
###############################################################################
## General ################################################

## Numerical - Histograms and Scatters #####################
def plotly_df_hists(
        df:         pd.DataFrame, 
        columns:    Optional[Iterable[str]] = None, 
        subplot:    bool = True,
        ncols:      Optional[int] = None,
        barmode:    str = 'group',
        title:      str = 'Bar charts',  
        **kwargs) -> None:
    '''Docstring of `plotly_df_hists`

    Plot histograms of DataFrame columns with plotly.

    Basicly same with `plotly_hists`.

    Args:
        df: A pandas DataFrame.
        columns: The column names. Optional.
            If omitted, plot all numerical columns.
        subplot: Whether to plot all data as subplots or not.
        ncols: Number of subplots of every row. 
            Ignored when `subplots` is `False`.
            If `None`, it's determined with `datasets`'s length.
        barmode: 'stack', 'group', 'overlay' or 'relative'.
            Ignored when `subplots` is `True`.
        title: Save the plot with this name.
    '''
    try:
        columns = list(columns)
    except TypeError:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    else:
        for col in columns:
            assert col in df, 'Column {} is not in given DataFrame.'.format(col)
    datasets = df[columns]

    return plotly_hists(datasets, orientation='vertical', names=columns, subplot=subplot, ncols=ncols, barmode=barmode, title=title, **kwargs)

def plotly_df_hist_grouped(
        df:         pd.DataFrame, 
        col:        str, 
        groupby:    str, 
        subplot:    bool = False,
        ncols:      Optional[int] = None, 
        barmode:    str = 'group',
        normdist:   bool = False, 
        title:      Optional[str] = None, 
        **kwargs) -> None:
    '''Docstring of `plotly_df_grouped_hist`

    Plot histograms of given column grouped on the unique value of another column with plotly.

    Args:
        df: A pandas DataFrame.
        col: Name of the column that the unique values of which
            will be used for grouping.
        groupby: Name of the column that will be grouped.
        subplot: Whether to plot all data as subplots or not.
        ncols: Number of subplots of every row.
        normdist: Whether plot the normal distribution with mean 
            and std of given data.
        title: Save the plot with this name.
    '''
    grouped = df.groupby(groupby)
    groups = grouped.groups.keys()
    datasets = [grouped.get_group(g)[col] for g in groups]
    return plotly_hists(datasets, names=groups, subplot=subplot, ncols=ncols, barmode=barmode, title=(title or 'Histogram {} grouped by {}'.format(col, groupby)))


''' normal distribution
    vals, data = grouped_col(df, col1, col2)
    nrows = int(np.ceil(len(vals) / ncols))
    fig = tls.make_subplots(rows=nrows, cols=ncols, subplot_titles=[str(v) for v in vals],
                            shared_yaxes=True, print_grid=False)
    width = kwargs.get('width', 900)
    height = kwargs.get('height', 700)
    # layout = go.Layout(title='Histogram of {} - grouped by {}'.format(col2, col1), showlegend=False, width=width, height=height)
    # for k in range(nrows):
    #     layout['yaxis{}'.format(k+1)]['title'] = 'Count'
    layout = {'yaxis{}'.format(k+1): {'title': 'Count'} for k in range(nrows)}
    layout.update(dict(
        title='Histogram of {} - grouped by {}'.format(col2, col1), 
        showlegend=False, width=width, height=height
    ))
    layout = go.Layout(layout)
    for i in range(nrows):
        for j in range(ncols):
            try:
                s = data[ncols * i + j]
            except:
                break
            start = s.min()
            end = s.max()
            bin_size = (end - start) / kwargs.get('bin_num', 10)
            if normdist:
                tempfig = ff.create_distplot([s], [s.name], curve_type='normal', histnorm='', bin_size=bin_size)
                for trace in tempfig['data']:
                    trace['xaxis'] = 'x{}'.format(i+1)
                    trace['yaxis'] = 'y{}'.format(j+1)
                    if trace['type'] == 'scatter':
                        trace['y'] = trace['y'] * s.count()
                    fig.append_trace(trace, i+1, j+1)
            else:
                trace = go.Histogram(x=s, xbins=dict(start=start, end=end, size=bin_size), name=s.name, autobinx=False)
                fig.append_trace(trace, i+1, j+1)
    fig['layout'].update(layout)
    kwargs.get('save', False) and plty.plot(fig, filename=title+'.html', image_width=width, image_height=height, auto_open=False)
    plty.iplot(fig)
'''

def plotly_df_boxes(
        df:         pd.DataFrame, 
        columns:    Optional[Iterable[str]] = None, 
        subplot:    bool = True,
        ncols:      Optional[int] = None,
        title:      str = 'Bar charts',  
        **kwargs) -> None:
    '''Docstring of `plotly_df_boxes`

    Plot box plots of DataFrame columns with plotly.

    Basicly same with `plotly_boxes`.

    Args:
        df: A pandas DataFrame.
        columns: The column names. Optional.
            If omitted, plot all numerical columns.
        subplot: Whether to plot all data as subplots or not.
        ncols: Number of subplots of every row. 
            Ignored when `subplots` is `False`.
            If `None`, it's determined with `datasets`'s length.
        title: Save the plot with this name.
    '''
    try:
        columns = list(columns)
    except TypeError:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    else:
        for col in columns:
            assert col in df, 'Column {} is not in given DataFrame.'.format(col)
    datasets = df[columns]

    return plotly_boxes(datasets, orientation='vertical', names=columns, subplot=subplot, ncols=ncols, title=title, **kwargs)
    
def plotly_df_box_grouped(
        df:         pd.DataFrame, 
        col:        str, 
        groupby:    str, 
        subplot:    bool = False,
        ncols:      Optional[int] = None,
        title:      Optional[str] = None,
        **kwargs) -> None:
    '''Docstring of `plotly_df_grouped_box`

    Plot box-plot of one column grouped by the unique value of another column with plotly.

    Args:
        df: A pandas DataFrame.
        col: Name of the column that the unique values of which
            will be used for grouping.
        groupby: Name of the column that will be grouped.
        title: Save the plot with this name.
    '''
    grouped = df.groupby(groupby)
    groups = grouped.groups.keys()
    datasets = [grouped.get_group(g)[col] for g in groups]
    return plotly_boxes(datasets, names=groups, subplot=subplot, ncols=ncols, title=(title or 'Histogram {} grouped by {}'.format(col, groupby)))
    # # df = df[[col1, col2]].dropna()
    # # cols = df[col1].unique()
    # # traces = [go.Box(y=df[df[col1] == col][col2], boxmean='sd', name=col) for col in cols]
    # vals, data = grouped_col(df, col1, col2)
    # traces = [go.Box(y=d, boxmean='sd', name=v) for v,d in zip(vals, data)]

    # width = kwargs.get('width', 900)
    # height = kwargs.get('height', 700)
    # layout = go.Layout(
    #     title='{} boxes grouped by {}'.format(col2, col1),
    #     yaxis=dict(title=col2),
    #     xaxis=dict(title=col1),
    #     width=width,
    #     height=height
    # )
    # fig = go.Figure(data=traces, layout=layout)
    # kwargs.get('save', False) and plty.plot(fig, filename=title+'.html', image_width=width, image_height=height, auto_open=False)
    # plty.iplot(fig)

def plotly_df_scatter(
        df:         pd.DataFrame, 
        colx:       str, 
        coly:       str, 
        size:       Union[str, float] = 6, 
        sizescale:  int = 100, 
        color:      str = None, 
        colorscale: Union[str, List[list]] = '10,div,RdYlBu', 
        title:      str = 'DataFrame Scatter',
        **kwargs) -> None:
    '''Docstring of `plotly_scatter`

    Plot scatter plot of two columns of given DataFrame with plotly.

    Basicly same with `plotly_scatter`.

    Args:
        df: A pandas DataFrame.
        colx: Name of column; x axis.
        coly: Name of column; y axis.
        size: Size of markers. 
            A column of given Dataframe, a float or a list.
            When passed a column name, the column will be used
            to determine the sizes of corresponding markers.
            If passed an iterable, it must be the same length 
            as datasets.
        sizescale: Ignored if `size` is a single value.
            A function to scale `size` to more reasonable values.
        color: Marker color. A column of given Dataframe, any
            kind of valid color string, or a numeric list.
            When passed a column name, the column will be used
            to determine the colors of corresponding markers.
            If list, the value will be used with `colorscale`.
        colorscale: Ignored when `color` is not a list.
            A list of `[percentage, color]` pairs, or a string
            that can be passed to `make_colorscale`.
        title: Save the plot with this name.
    '''
    assert colx in df, 'Column {} not in given dataframe'.format(colx)
    assert coly in df, 'Column {} not in given dataframe'.format(coly)
    x = df[colx].values
    y = df[coly].values
    if isinstance(size, str) and size in df:
        size = df[size]
    if isinstance(color, str) and color in df:
        color = df[color]
    return plotly_scatter(x, y, size, sizescale, color, colorscale, title, **kwargs)
    
def plotly_df_scatter_matrix(
        df:         pd.DataFrame,
        columns:    Optional[List[str]] = None,
        size:       Union[str, float] = 2, 
        sizescale:  int = 100, 
        color:      str = None, 
        colorscale: Union[str, list] = '10,div,RdYlBu',
        title:      str = 'Scatter Matrix',
        **kwargs) -> None:
    '''Docstring of `plotly_df_scatter`

    Plot scatter matrix of any number of given columns with plotly.

    Basicly same with `plotly_scatter_matrix`.

    Args:
        df: A pandas DataFrame.
        columns: Columns' names for plotting.
            If `None`, use all numerical columns.
        size: Marker size. A column of given Dataframe, a float,
            or a numerical list.
            When passed a column name, the column will be used
            to determine the sizes of corresponding markers.
            If a list, it must be the same length as datasets.
        sizescale: Ignored when `size` is a float.
            Scale the Dataframe to fit markers' size to the plot.
        color: Marker color. A column of given Dataframe, any
            kind of valid color string, or a numerical list.
            When passed a column name, the column will be used
            to determine the colors of corresponding markers.
            If list, the value will be used with `colorscale`.
        colorscale: Ignored when `color` is not a column of 
            given Dataframe.
            A list of `[percentage, color]` pairs, or a string
            that can be passed to `make_colorscale`.
        title: Save the plot with this name.    
    '''
    try:
        columns = list(columns)
    except TypeError:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    else:
        for col in columns:
            assert col in df, 'Column {} is not in given DataFrame.'.format(col)
    if isinstance(size, str) and size in df:
        size = df[size]
    if isinstance(color, str) and color in df:
        color = df[color]
    
    datasets = df[columns]
    return plotly_scatter_matrix(datasets, orientation='vertical', names=columns, size=size, sizescale=sizescale, color=color, colorscale=colorscale, title=title, **kwargs)

def plotly_df_clusters(
        df:         pd.DataFrame, 
        colx:       str, 
        coly:       str, 
        collabel:   str, 
        names:      dict = None,
        sizes=6,
        colorscale: Union[str, list] = '12,qual,Paired', 
        title:      str = 'DataFrame 2D Clusters',  
        **kwargs) -> None:
    '''Docstring of `plotly_df_2d_clusters`

    Plot scatter plots of two columns grouped by the unique values
    of a third column with plotly.

    Basicly same with `plotly_clusters`.

    Args:
        df: A pandas DataFrame.
        colx: The column name of x values.
        coly: The column name of y values.
        collabel: The column name to group x and y on which.
        names: Lables' names. Optional.
        sizes: A single number or a number list.
            Marker sizes of each cluster.
        colorscale: Colors of clusters. 
            A list of `[percentage, color]` pairs, or a string
            that can be passed to `make_colorscale`.
        title: Title of the plot.
    '''
    assert colx in df, 'Column {} not in given dataframe'.format(colx)
    assert coly in df, 'Column {} not in given dataframe'.format(coly)
    assert collabel in df, 'Column {} not in given dataframe'.format(collabel)
    x = df[colx].values
    y = df[coly].values
    label = df[collabel].values

    return plotly_clusters(x, y, label, names=names, sizes=sizes, colorscale=colorscale, title=title, **kwargs)

def plotly_df_qq_plots(
        df:         pd.DataFrame,
        columns:    Optional[List[str]] = None,
        ncols:      Optional[int] = None,
        title:      str = 'DaraFrame QQ plots', 
        **kwargs) -> None:
    '''Docstring of `plotly_qq_plots`

    Plot QQ-plots of all the given data with plotly.

    Args:
        datasets: A list of data to plot.
        names: Data names corresponding to items in data. Optional.
        ncols: Number of subplots of every row. 
            Ignored when `subplots` is `False`.
            If `None`, it's determined with `datasets`'s length.
        title: Save the plot with this name.
    '''
    try:
        columns = list(columns)
    except TypeError:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    else:
        for col in columns:
            assert col in df, 'Column {} is not in given DataFrame.'.format(col)
    
    datasets = df[columns]
    return plotly_qq_plots(datasets, orientation='vertical', names=columns, ncols=ncols, title=title, **kwargs)
    
## Categorical - Bars and contingency tables #####################
def plotly_df_bars(
        df:         pd.DataFrame, 
        columns:    Optional[Iterable[str]] = None, 
        subplot:    bool = False,
        ncols:      Optional[int] = None,
        barmode:    str = 'group',
        title:      str = 'Bar charts', 
        **kwargs) -> None:
    '''Docstring of `plotly_df_bars`

    Plot bar charts of DataFrame columns with plotly.

    Basicly same with `plotly_bars`.

    Args:
        df: A pandas DataFrame.
        columns: The column names. Optional.
            If omitted, plot all categorical columns.
        subplot: Whether to plot all data as subplots or not.
        ncols: Number of subplots of every row. 
            Ignored when `subplots` is `False`.
            If `None`, it's determined with `datasets`'s length.
        barmode: 'stack', 'group', 'overlay' or 'relative'.
            Ignored when `subplots` is `True`.
        title: Save the plot with this name.
    '''
    try:
        columns = list(columns)
    except TypeError:
        columns = df.select_dtypes(include=[object]).columns.tolist()
    else:
        for col in columns:
            assert col in df, 'Column {} is not in given DataFrame.'.format(col)
    datasets = df[columns]
    return plotly_bars(datasets, orientation='vertical', names=columns, subplot=subplot, ncols=ncols, barmode=barmode, title=title, **kwargs)

def plotly_df_crosstable(
        df:         pd.DataFrame, 
        columns:    Optional[Iterable[str]] = None, 
        half:       bool = False,
        ttype:      str = 'count', 
        title:      str = 'DataFrame Contigency Table',
        **kwargs) -> None:
    '''Docstring of `plotly_df_crosstable`

    Plot contigency table of two given columns with plotly heatmap.

    Basicly same with `plotly_crosstable`

    Args:
        df: A pandas DataFrame.
        columns: The column names. Optional.
            If omitted, plot all categorical columns.
        half: Stacked bar plot or heatmap.
        ttype: Determines how the contigency table is calculated.
            'count': The counts of every combination.
            'percent': The percentage of every combination to the 
            sum of every rows.
            Defaults to 'count'.
        colorscale: Heatmap colorscale. Avaiable values are 
            'Blackbody', 'Bluered', 'Blues', 'Earth', 'Electric', 'Greens',
            'Greys', 'Hot', 'Jet', 'Picnic', 'Portland', 'Rainbow', 'RdBu',
            'Reds', 'Viridis', 'YlGnBu', 'YlOrRd'.
        title: Save the plot with this name.
    '''
    try:
        columns = list(columns)
    except TypeError:
        columns = df.select_dtypes(include=[object]).columns.tolist()
    else:
        for col in columns:
            assert col in df, 'Column {} is not in given DataFrame.'.format(col)
    datasets = df[columns]
    return plotly_crosstable(datasets, orientation='vertical', names=columns, half=half, ttype=ttype, title=title, **kwargs)

def plotly_df_crosstable_stacked(
        df:         pd.DataFrame, 
        columns:    Optional[Iterable[str]] = None, 
        half:       bool = False,
        ttype:      str = 'count', 
        title:      str = 'DataFrame Contigency Table',
        **kwargs) -> None:
    '''Docstring of `plotly_df_crosstable_stacked`

    Plot contigency table of two given columns with plotly heatmap.

    Basicly same with `plotly_crosstable_stacked`

    Args:
        df: A pandas DataFrame.
        columns: The column names. Optional.
            If omitted, plot all categorical columns.
        half: Stacked bar plot or heatmap.
        ttype: Determines how the contigency table is calculated.
            'count': The counts of every combination.
            'percent': The percentage of every combination to the 
            sum of every rows.
            Defaults to 'count'.
        colorscale: Heatmap colorscale. Avaiable values are 
            'Blackbody', 'Bluered', 'Blues', 'Earth', 'Electric', 'Greens',
            'Greys', 'Hot', 'Jet', 'Picnic', 'Portland', 'Rainbow', 'RdBu',
            'Reds', 'Viridis', 'YlGnBu', 'YlOrRd'.
        title: Save the plot with this name.
    '''
    try:
        columns = list(columns)
    except TypeError:
        columns = df.select_dtypes(include=[object]).columns.tolist()
    else:
        for col in columns:
            assert col in df, 'Column {} is not in given DataFrame.'.format(col)
    datasets = df[columns]
    return plotly_crosstable_stacked(datasets, orientation='vertical', names=columns, half=half, ttype=ttype, title=title, **kwargs)

def plotly_df_chi_square_matrix(
        df:             pd.DataFrame, 
        columns:        Optional[Iterable[str]] = None, 
        title:          str = 'Chi-Square Matrix',
        **kwargs) -> None:
    '''Docstring of `plotly_df_chi_square_matrix`

    Run chi-square test on every two columns of given pandas DataFrame,
    with contigency tables calculated on the two columns.
    Then plot the results as a matrix.

    Args:
        df: A pandas DataFrame.
        columns: The column names. Optional.
            If omitted, plot all categorical columns.
        title: Save the plot with this name.
    '''
    try:
        columns = list(columns)
    except TypeError:
        columns = df.select_dtypes(include=[object]).columns.tolist()
    else:
        for col in columns:
            assert col in df, 'Column {} is not in given DataFrame.'.format(col)
    datasets = df[columns]
    return plotly_chi_square_matrix(datasets, orientation='vertical', names=columns, title=title, **kwargs)



###############################################################################
## Plotly Array plot functions
###############################################################################
## General ################################################
def plotly_describes(
        data:   list, 
        names:  list = [], 
        title:  str = 'Descriptive Statistics',
        **kwargs) -> None:
    '''Docstring of `plotly_describes`

    Plot a table of descriptive statistics of given data with plotly.

    Args:
        data: A list of numerical data.
        names: A list contains names corresponding to data.
        title: Save the plot with this name.
    '''
    ndata = len(data)
    names = names or ['']*ndata
    describes = np.empty((12, ndata+1), dtype=object)
    for i, d, n in zip(range(ndata), data, names):
        des = advanced_describe(d, name=n)
        if i == 0:
            describes[0, 0] = 'Describes'
            describes[1:, 0] = des[2:, 0]
        describes[0, i+1] = des[0, 1]
        describes[1:, i+1] = ['{:.2f}'.format(float(v)) for v in des[2:, 1]]
    fig = ff.create_table(describes, index=True)
    width = kwargs.get('width', 900)
    height = kwargs.get('height', 700)
    fig.layout.width = width
    fig.layout.height = height
    fig.layout.title = title
    kwargs.get('save', False) and plty.plot(fig, filename=title+'.html', image_width=width, image_height=height, auto_open=False)
    if kwargs.get('return_fig', False):
        return fig
    else:
        plty.iplot(fig)

## Numerical - Histograms and Scatters #####################
def plotly_hists(
        datasets:   Union[list, np.array], 
        names:      Optional[Iterable[str]] = None, 
        subplot:    bool = False,
        ncols:      Optional[int] = None,
        barmode:    str = 'group',
        title:      str = 'Histogram', 
        **kwargs) -> None:
    '''Docstring of `plotly_hists`

    Plot histograms with plotly.

    Args:
        datasets: A list of data to plot.
        names: Data names corresponding to items in data. Optional.
        subplot: Whether to plot all data as subplots or not.
        ncols: Number of subplots of every row. 
            Ignored when `subplots` is `False`.
            If `None`, it's determined with `datasets`'s length.
        barmode: 'stack', 'group', 'overlay' or 'relative'.
            Ignored when `subplots` is `True`.
        title: Save the plot with this name.
    '''
    width = kwargs.get('width', 900)
    height = kwargs.get('height', 700)
    layout = go.Layout(title=title, width=width, height=height)
    layout.update(kwargs.get('layout', {}))
    # preparation
    datasets = np.array(datasets)
    if kwargs.get('orientation', 'horizontal') == 'vertical':
        datasets = np.array(datasets).T
    ndata = datasets.shape[0]
    if names is not None:
        assert len(names) == ndata, 'Not enough names for data length {}. Given {}.'.format(ndata, len(names))
    else:
        names = ['trace{}'.format(i+1) for i in range(ndata)]
    hist_common_layout = kwargs.get('hist_common_layout', {})
    hist_unique_layouts = kwargs.get('hist_unique_layout', [{}]*ndata)

    # make traces for every data row
    traces = []
    for data, name, hist_layout in zip(datasets, names, hist_unique_layouts):
        start = np.min(data)
        end = np.max(data)
        bin_size = np.ptp(data) / kwargs.get('bin_num', 10)
        if kwargs.get('horizontal', False):
            trace = go.Histogram(y=data, xbins=dict(start=start, end=end, size=bin_size), name=name, **hist_layout, **hist_common_layout)
        else:
            trace = go.Histogram(x=data, xbins=dict(start=start, end=end, size=bin_size), name=name, **hist_layout, **hist_common_layout)
        traces.append(trace)
    
    if subplot:
        if ncols is None:
            nrows = int(np.floor(np.power(ndata, 0.5)))
            ncols = int(np.ceil(ndata / nrows))
        else:
            nrows = int(np.ceil(ndata / ncols))
        fig = tls.make_subplots(rows=nrows, cols=ncols, subplot_titles=names, 
                                vertical_spacing=0.1, horizontal_spacing=0.1, print_grid=False)
        for i, trace in enumerate(traces):
            fig.append_trace(trace, i // ncols + 1, i % ncols + 1)
    else:
        if ndata > 1:
            assert barmode in ['stack', 'group', 'overlay', 'relative'], 'Invalid barmode "{}".'.format(barmode)
            layout.barmode = barmode
        fig = go.Figure(data=traces)
    
    fig['layout'].update(layout)
    kwargs.get('save', False) and plty.plot(fig, filename=title+'.html', image_width=width, image_height=height, auto_open=False)
    if kwargs.get('return_fig', False):
        return fig
    else:
        plty.iplot(fig)

def plotly_boxes(
        datasets:   Union[list, np.array], 
        names:      Optional[Iterable[str]] = None, 
        subplot:    bool = False,
        ncols:      Optional[int] = None,
        title:      str = 'Histogram', 
        **kwargs) -> None:
    '''Docstring of `plotly_hists`

    Plot histograms with plotly.

    Args:
        datasets: A list of data to plot.
        names: Data names corresponding to items in data. Optional.
        subplot: Whether to plot all data as subplots or not.
        ncols: Number of subplots of every row. 
            Ignored when `subplots` is `False`.
            If `None`, it's determined with `datasets`'s length.
        title: Save the plot with this name.
    '''
    width = kwargs.get('width', 900)
    height = kwargs.get('height', 700)
    layout = go.Layout(title=title, width=width, height=height)
    layout.update(kwargs.get('layout', {}))
    # preparation
    datasets = np.array(datasets)
    if kwargs.get('orientation', 'horizontal') == 'vertical':
        datasets = np.array(datasets).T
    ndata = datasets.shape[0]
    if names is not None:
        assert len(names) == ndata, 'Not enough names for data length {}. Given {}.'.format(ndata, len(names))
    else:
        names = ['trace{}'.format(i+1) for i in range(ndata)]
    box_common_layout = kwargs.get('box_common_layout', {})
    box_unique_layouts = kwargs.get('box_unique_layout', [{}]*ndata)
    
    # make traces for every data row
    traces = []
    for data, name, box_layout in zip(datasets, names, box_unique_layouts):
        if kwargs.get('horizontal', False):
            trace = go.Box(x=data, boxmean='sd', name=name, **box_layout, **box_common_layout)
        else:
            trace = go.Box(y=data, boxmean='sd', name=name, **box_layout, **box_common_layout)
        traces.append(trace)
    
    if subplot:
        if ncols is None:
            nrows = int(np.floor(np.power(ndata, 0.5)))
            ncols = int(np.ceil(ndata / nrows))
        else:
            nrows = int(np.ceil(ndata / ncols))
        fig = tls.make_subplots(rows=nrows, cols=ncols, subplot_titles=names, 
                                vertical_spacing=0.1, horizontal_spacing=0.1, print_grid=False)
        for i, trace in enumerate(traces):
            fig.append_trace(trace, i // ncols + 1, i % ncols + 1)
    else:
        fig = go.Figure(data=traces)
    
    fig['layout'].update(layout)
    kwargs.get('save', False) and plty.plot(fig, filename=title+'.html', image_width=width, image_height=height, auto_open=False)
    if kwargs.get('return_fig', False):
        return fig
    else:
        plty.iplot(fig)

def plotly_scatter(
        x,
        y,
        size = 6, 
        sizescale:  Optional[callable] = None,
        color = None, 
        colorscale: Union[str, List[list]] = '10,div,RdYlBu',
        title:      str = 'Scatter', 
        **kwargs) -> None:
    '''Docstring of `plotly_scatter`

    Plot scatters with plotly.

    Args:
        x: x axis.
        y: y axis. `x` and `y` must have same length.
        size: Size of markers. If passed an iterable, it must be 
            the same length as datasets.
        sizescale: Ignored if `size` is a single value.
            A function to scale `size` to more reasonable values.
        color: Marker color. A numeric list or any
            kind of valid color string.
            If list, the value will be used with `colorscale`.
        colorscale: Ignored when `color` is not a list.
            A list of `[percentage, color]` pairs, or a string
            that can be passed to `make_colorscale`.
        title: Save the plot with this name.
    '''
    assert len(x) == len(y), 'Given data have different length. `x`:{}, `y`:{}'.format(len(x), len(y))
    width = kwargs.get('width', 900)
    height = kwargs.get('height', 700)
    layout = go.Layout(title=title, width=width, height=height)
    layout.update(kwargs.get('layout', {}))

    if not isinstance(size, (int, float)):
        sizescale = sizescale or (lambda array: (np.array(array) - np.min(array)) / np.ptp(array) * 30)
        size = sizescale(size)
    marker = dict(
        opacity     = kwargs.get('marker_opacity', 0.5), 
        size        = size, 
        showscale   = False,
        line        = dict(
            width = kwargs.get('marker_line_width', 1),
            color = kwargs.get('marker_line_color', 'rgb(0,0,0)')
        )
    )
    if color is not None:
        if not isinstance(color, str):
            if isinstance(colorscale, str):
                colorscale =  make_colorscale(colorscale, reverse=kwargs.get('colorscale_reverse', False))
            marker['colorscale'] = colorscale
            marker['showscale'] = kwargs.get('showscale', True)
        marker['color'] = color
    trace = go.Scattergl(
        x=x, y=y, mode='markers',
        marker=marker
    )

    fig = go.Figure(data=[trace])
    fig['layout'].update(layout)
    kwargs.get('save', False) and plty.plot(fig, filename=title+'.html', image_width=width, image_height=height, auto_open=False)
    if kwargs.get('return_fig', False):
        return fig
    else:
        plty.iplot(fig)

def plotly_scatter_matrix(
        datasets,
        names:      Optional[List[str]] = None,
        size = 6, 
        sizescale:  Optional[callable] = None,
        color = None, 
        colorscale: Union[str, List[list]] = '10,div,RdYlBu',
        title:      str = 'Scatter Matrix',
        **kwargs) -> None:
    '''Docstring of `plotly_scatter_matrix`

    Plot scatter matrix of any set of datasets with plotly.

    The subplots from top left to bottom right on diagonal are replaced with histograms.

    Args:
        datasets: A list of data to plot.
        names: Names corresponding to items in datasets. Optional.
        size: Size of markers. If passed an iterable, it must be 
            the same length as datasets.
        sizescale: Ignored if `size` is a single value.
            A function to scale `size` to more reasonable values.
        color: Marker color. A numeric list or any
            kind of valid color string.
            If list, the value will be used with `colorscale`.
        colorscale: Ignored when `color` is not a list.
            A list of `[percentage, color]` pairs, or a string
            that can be passed to `make_colorscale`.
        title: Save the plot with this name.    
    '''
    width = kwargs.get('width', 900)
    height = kwargs.get('height', 700)
    layout = go.Layout(title=title, width=width, height=height)
    layout.update(kwargs.get('layout', {}))
    # preparation
    datasets = np.array(datasets)
    if kwargs.get('orientation', 'horizontal') == 'vertical':
        datasets = np.array(datasets).T
    ndata = datasets.shape[0]
    if names is not None:
        assert len(names) == ndata, 'Not enough names for data length {}. Given {}.'.format(ndata, len(names))
    else:
        names = ['trace{}'.format(i+1) for i in range(ndata)]
    hist_common_layout = kwargs.get('hist_common_layout', {})
    hist_unique_layouts = kwargs.get('hist_unique_layout', [{}]*ndata)

    if not isinstance(size, (int, float)):
        sizescale = sizescale or (lambda array: (np.array(array) - np.min(array)) / np.ptp(array) * 30)
        size = sizescale(size)
    marker = dict(
        opacity     = kwargs.get('marker_opacity', 0.5), 
        size        = size, 
        showscale   = False,
        line        = dict(
            width = kwargs.get('marker_line_width', 1),
            color = kwargs.get('marker_line_color', 'rgb(0,0,0)')
        )
    )
    if color is not None:
        if not isinstance(color, str):
            if isinstance(colorscale, str):
                colorscale =  make_colorscale(colorscale, reverse=kwargs.get('colorscale_reverse', False))
            marker['colorscale'] = colorscale
            marker['showscale'] = kwargs.get('showscale', True)
        marker['color'] = color

    # make trace
    trace = go.Splom(
        dimensions=[{'label': name, 'values': data} for name, data in zip(names, datasets)],
        marker=marker,
        diagonal={'visible': False}
    )
    traces = [trace]
    # add histogram
    for i, (name, data, hist_layout) in enumerate(zip(names, datasets, hist_unique_layouts)):
        start = np.min(data)
        end = np.max(data)
        bin_size = np.ptp(data) / kwargs.get('bin_num', 10)
        trace = go.Histogram(
            x=data, xbins=dict(start=start, end=end, size=bin_size), 
            name=name, autobinx=False,
            xaxis='x{}'.format(i+ndata+1),
            yaxis='y{}'.format(i+ndata+1,
            **hist_layout, **hist_common_layout)
        )
        traces.append(trace)
    hist_pos = kwargs.get('hist_pos', 0.15)
    layout.update({
        'xaxis{}'.format(i+ndata+1): {'domain': [1-(i+1)/ndata+hist_pos/ndata, 1-i/ndata-hist_pos/ndata], 'anchor': 'x{}'.format(i+ndata+1)}
    for i in range(ndata)})
    layout.update({
        'yaxis{}'.format(i+ndata+1): {'domain': [i/ndata+hist_pos/ndata, (i+1)/ndata-hist_pos/ndata], 'anchor': 'y{}'.format(i+ndata+1)}
    for i in range(ndata)})
    layout.update(dict(
        dragmode=kwargs.get('dragmode', 'select'),
        hovermode='closest',
        plot_bgcolor=kwargs.get('plot_bgcolor', None),
        showlegend=False
    ))
    fig = go.Figure(data=traces, layout=layout)
    kwargs.get('save', False) and plty.plot(fig, filename=title+'.html', image_width=width, image_height=height, auto_open=False)
    if kwargs.get('return_fig', False):
        return fig
    else:
        plty.iplot(fig)

def plotly_clusters(
        x,
        y,
        label,
        names:      dict = None,
        sizes=6,
        colorscale: Union[str, list] = '12,qual,Paired', 
        title:      str = '2D Clusters', 
        **kwargs) -> None:
    '''Docstring of `plotly_clusters`

    Plot 2D clusters with plotly.

    Args:
        x: x axis.
        y: y axis. 
        label: Label of data for clustering. 
            `x`, `y` and `label` must have same length.
        names: Lables' names. Optional.
        sizes: A single number or a number list.
            Marker sizes of each cluster.
        colorscale: Colors of clusters. 
            A list of `[percentage, color]` pairs, or a string
            that can be passed to `make_colorscale`.
        title: Title of the plot.
    '''
    assert len(x) == len(y) == len(label), 'Given data have different length. `x`:{}, `y`:{}, `label`:{}'.format(len(x), len(y), len(label))
    width = kwargs.get('width', 900)
    height = kwargs.get('height', 700)
    layout = go.Layout(title=title, width=width, height=height, hovermode='closest')
    layout.update(kwargs.get('layout', {}))

    # group by labels
    labels = np.unique(label)
    datasets = np.c_[x, y, label]
    if names is not None:
        datasets = {names[k]: datasets[datasets[:,2]==k] for k in labels}
        labels = [names[k] for k in labels]
    else:
        datasets = {k: datasets[datasets[:,2]==k] for k in labels}
    n_labels = len(labels)
    # check for size and colors
    if isinstance(sizes, (int, float)):
        sizes = [sizes]*n_labels
    if isinstance(colorscale, str):
        colorscale = [c[1] for c in make_colorscale(colorscale, n=n_labels, reverse=kwargs.get('colorscale_reverse', False))]
    active_opacity = kwargs.get('active_opacity', 1)
    active_line_width = kwargs.get('active_line_width', 1)
    inactive_opacity = kwargs.get('inactive_opacity', 0.05)
    inactive_line_width = kwargs.get('inactive_line_width', 0.1)
    # buttons for selecting cluster to highlight
    buttons = [dict(
        label = 'Reset',
        method = 'restyle',
        args = [{
            'marker.opacity': [active_opacity] * n_labels,
            'marker.line.width': [active_line_width]*n_labels
        }]
    )]
    # add data points cluster by cluster
    traces = []
    for i, (label, size, color) in enumerate(zip(labels, sizes, colorscale)):
        data = datasets[label]
        trace = go.Scattergl(
            x=data[:,0], y=data[:,1], mode='markers', name=label, 
            marker=dict(color=color, size=size, opacity=active_opacity, line=dict(width=active_line_width)))
        traces.append(trace)
        button = dict(
            label = 'Cluster {}'.format(label),
            method = 'restyle',
            args = [{
                'marker.opacity': [inactive_opacity]*i + [active_opacity] + [inactive_opacity]*(n_labels-i-1),
                'marker.line.width': [inactive_line_width]*i + [active_line_width] + [inactive_line_width]*(n_labels-i-1),
            }]
        )
        buttons.append(button)
    updatemenus = [
        dict(
            type='buttons',
            buttons=buttons
        )
    ]
    layout.updatemenus = updatemenus
    fig = go.Figure(data=traces, layout=layout)
    kwargs.get('save', False) and plty.plot(fig, filename=title+'.html', image_width=width, image_height=height, auto_open=False)
    if kwargs.get('return_fig', False):
        return fig
    else:
        plty.iplot(fig)

def plotly_qq_plots(
        datasets:   Union[list, np.array], 
        names:      Optional[Iterable[str]] = None, 
        ncols:      Optional[int] = None,
        title:      str = 'QQ plots', 
        **kwargs) -> None:
    '''Docstring of `plotly_qq_plots`

    Plot QQ-plots of all the given data with plotly.

    Args:
        datasets: A list of data to plot.
        names: Data names corresponding to items in data. Optional.
        ncols: Number of subplots of every row. 
            Ignored when `subplots` is `False`.
            If `None`, it's determined with `datasets`'s length.
        title: Save the plot with this name.
    '''
    width = kwargs.get('width', 900)
    height = kwargs.get('height', 700)
    layout = go.Layout(title=title, width=width, height=height)
    layout.update(kwargs.get('layout', {}))
    # preparation
    datasets = np.array(datasets)
    if kwargs.get('orientation', 'horizontal') == 'vertical':
        datasets = np.array(datasets).T
    ndata = datasets.shape[0]
    if names is not None:
        assert len(names) == ndata, 'Not enough names for data length {}. Given {}.'.format(ndata, len(names))
    else:
        names = ['trace{}'.format(i+1) for i in range(ndata)]
    if ncols is None:
        nrows = int(np.floor(np.power(ndata, 0.5)))
        ncols = int(np.ceil(ndata / nrows))
    else:
        nrows = int(np.ceil(ndata / ncols))
    fig = tls.make_subplots(rows=nrows, cols=ncols, subplot_titles=names, 
                            vertical_spacing=0.1, horizontal_spacing=0.1, print_grid=False)
    probs = [stats.probplot(data if len(data) <= 1000 else np.random.choice(data, 1000)) for data in datasets]
    
    for i, prob in enumerate(probs):
        fig.append_trace(go.Scattergl(x=prob[0][0], y=prob[0][1], mode='markers', name=names[i]+'_real'), i // ncols + 1, i % ncols + 1)
        fig.append_trace(go.Scattergl(x=prob[0][0], y=prob[0][0]*prob[1][0]+prob[1][1], mode='lines', name=names[i]+'_exp'), i // ncols + 1, i % ncols + 1)

    fig['layout'].update(layout)
    kwargs.get('save', False) and plty.plot(fig, filename=title+'.html', image_width=width, image_height=height, auto_open=False)
    if kwargs.get('return_fig', False):
        return fig
    else:
        plty.iplot(fig)

## Categorical - Bars and contingency tables #####################
def plotly_bars(
        datasets, 
        names:      Optional[Iterable[str]] = None, 
        subplot:    bool = False,
        ncols:      Optional[int] = None,
        barmode:    str = 'group',
        title:      str = 'Bar charts', 
        **kwargs) -> None:
    '''Docstring of `plotly_bars`

    Plot bar charts with plotly.

    Args:
        datasets: A list of data to plot.
        names: Data names corresponding to items in data. Optional.
        subplot: Whether to plot all data as subplots or not.
        ncols: Number of subplots of every row. 
            Ignored when `subplots` is `False`.
            If `None`, it's determined with `datasets`'s length.
        barmode: 'stack', 'group', 'overlay' or 'relative'.
            Ignored when `subplots` is `True`.
        title: Save the plot with this name.
    '''
    width = kwargs.get('width', 900)
    height = kwargs.get('height', 700)
    layout = go.Layout(title=title, width=width, height=height)
    layout.update(kwargs.get('layout', {}))
    # preparation
    datasets = np.array(datasets)
    if kwargs.get('orientation', 'horizontal') == 'vertical':
        datasets = np.array(datasets).T
    ndata = datasets.shape[0]
    if names is not None:
        assert len(names) == ndata, 'Not enough names for data length {}. Given {}.'.format(ndata, len(names))
    else:
        names = ['trace{}'.format(i+1) for i in range(ndata)]
    bar_common_layout = kwargs.get('bar_common_layout', {})
    bar_unique_layouts = kwargs.get('bar_unique_layout', [{}]*ndata)
    
    # make traces for every data row
    traces = []
    for data, name, bar_layout in zip(datasets, names, bar_unique_layouts):
        s = pd.value_counts(data).sort_index()
        trace = go.Bar(x=s.index, y=s.values, text=s.values, textposition='auto', 
            name=name, **bar_layout, **bar_common_layout)
        traces.append(trace)
    
    if subplot:
        if ncols is None:
            nrows = int(np.floor(np.power(ndata, 0.5)))
            ncols = int(np.ceil(ndata / nrows))
        else:
            nrows = int(np.ceil(ndata / ncols))
        fig = tls.make_subplots(rows=nrows, cols=ncols, subplot_titles=names, 
                                vertical_spacing=0.1, horizontal_spacing=0.1, print_grid=False)
        for i, trace in enumerate(traces):
            fig.append_trace(trace, i // ncols + 1, i % ncols + 1)
    else:
        if ndata > 1:
            assert barmode in ['stack', 'group', 'overlay', 'relative'], 'Invalid barmode "{}".'.format(barmode)
            layout.barmode = barmode
        fig = go.Figure(data=traces)

    fig['layout'].update(layout)
    kwargs.get('save', False) and plty.plot(fig, filename=title+'.html', image_width=width, image_height=height, auto_open=False)
    if kwargs.get('return_fig', False):
        return fig
    else:
        plty.iplot(fig)

def plotly_crosstable(
        datasets,
        names=None,
        half:       bool = False,
        ttype:      str = 'count',
        colorscale: Union[str, list] = 'Greys', 
        title:      str = 'Contigency Table',
        **kwargs) -> None:
    '''Docstring of `plotly_crosstable`

    Plot contigency table (matrix) with plotly heatmap.

    Args:
        datasets: A list of data to plot.
        names: Data names correspond to columns of datasets. Optional.
        half: Shows only half of the matrix or shows duplicated part too.
        ttype: Determines how the contigency table is calculated.
            'count': The counts of every combination.
            'percent': The percentage of every combination to the 
            sum of every rows.
            Defaults to 'count'.
        colorscale: Heatmap colorscale. Avaiable values are 
            'Blackbody', 'Bluered', 'Blues', 'Earth', 'Electric', 'Greens',
            'Greys', 'Hot', 'Jet', 'Picnic', 'Portland', 'Rainbow', 'RdBu',
            'Reds', 'Viridis', 'YlGnBu', 'YlOrRd'.
        title: Save the plot with this name.
    '''
    width = kwargs.get('width', 900)
    height = kwargs.get('height', 700)
    # preparation
    datasets = np.array(datasets)
    if kwargs.get('orientation', 'horizontal') == 'vertical':
        datasets = np.array(datasets).T
    ndata = datasets.shape[0]
    if names is not None:
        assert len(names) == ndata, 'Not enough names for data length {}. Given {}.'.format(ndata, len(names))
    else:
        names = ['trace{}'.format(i+1) for i in range(ndata)]
    if half:
        range1, range2 = range(ndata-1), range(1, ndata)
    else:
        range1, range2 = range(ndata), range(ndata)
    
    # layout
    layout = go.Layout(title=title, annotations=[], width=width, height=height)
    layout.update({'xaxis{}'.format(k+(not half)): {'title': names[k], 'type': 'category'} for k in range2})
    layout.update({'yaxis{}'.format(k+1): {'title': names[k], 'type': 'category', 'autorange': 'reversed'} for k in range1})
    
    # make fig for subplots
    fig = tls.make_subplots(rows=len(range2), cols=len(range1), 
                            shared_xaxes=True, shared_yaxes=True, 
                            vertical_spacing=0.01, horizontal_spacing=0.01, print_grid=False)
    for i in range1:
        for j in range2:
            if half and i == j:
                continue
            ct = contingency_table(datasets[i], datasets[j], rownames=[names[i]], colnames=[names[j]])
            if ttype == 'percent':
                total_row = ct.iloc[-1] / ct.iloc[-1, -1]
                ct.index = ct.index.astype(str) + ' (' + ct['Total'].astype(str) + ')'
                ct.columns = ct.columns.astype(str) + ' (' + ct.iloc[-1].astype(str) + ')'
                ct = ct / ct.iloc[-1]
                ct.iloc[-1] = total_row.values
                ct = ct * 100
            annheat = ff.create_annotated_heatmap(
                z=ct.values, x=list(ct.columns), y=list(ct.index),
                annotation_text=(None if ttype=='count' else ct.applymap('{:.2f}%'.format).values),
                colorscale=colorscale,
                hoverinfo='x+y',
            )
            trace = annheat['data'][0]
            idx_i = i + 1
            idx_j = j + (not half)
            
            annotations = annheat['layout']['annotations']
            for ann in annotations:
                ann['xref'] = 'x{}'.format(idx_j)
                ann['yref'] = 'y{}'.format(idx_i)
            # print(layout['annotations'])
            layout['annotations'].extend(annotations)
            fig.append_trace(trace, idx_i, idx_j)
    
    layout.update(kwargs.get('layout', {}))
    fig['layout'].update(layout)
    kwargs.get('save', False) and plty.plot(fig, filename=title+'.html', image_width=width, image_height=height, auto_open=False)
    if kwargs.get('return_fig', False):
        return fig
    else:
        plty.iplot(fig)

def plotly_crosstable_stacked(
        datasets,
        names=None,
        half:       bool = False,
        ttype:      str = 'count',
        colorscale: Union[str, list] = '3,seq,Greys', 
        title:      str = 'Stacked Contigency Table',
        **kwargs) -> None:
    '''Docstring of `plotly_crosstable`

    Plot stacked contigency table with plotly heatmap.

    Args:
        datasets: A list of data to plot.
        names: Data names correspond to columns of datasets. Optional.
        half: Shows only half of the matrix or shows duplicated part too.
        ttype: Determines how the contigency table is calculated.
            'count': The counts of every combination.
            'percent': The percentage of every combination to the 
            sum of every rows.
            Defaults to 'count'.
        colorscale: Heatmap colorscale. Avaiable values are 
            'Blackbody', 'Bluered', 'Blues', 'Earth', 'Electric', 'Greens',
            'Greys', 'Hot', 'Jet', 'Picnic', 'Portland', 'Rainbow', 'RdBu',
            'Reds', 'Viridis', 'YlGnBu', 'YlOrRd'.
        title: Save the plot with this name.
    '''
    width = kwargs.get('width', 900)
    height = kwargs.get('height', 700)
    # preparation
    datasets = np.array(datasets)
    if kwargs.get('orientation', 'horizontal') == 'vertical':
        datasets = np.array(datasets).T
    ndata = datasets.shape[0]
    if names is not None:
        assert len(names) == ndata, 'Not enough names for data length {}. Given {}.'.format(ndata, len(names))
    else:
        names = ['trace{}'.format(i+1) for i in range(ndata)]
    if half:
        range1, range2 = range(ndata-1), range(1, ndata)
    else:
        range1, range2 = range(ndata), range(ndata)

    # layout
    layout = go.Layout(
        title=title, annotations=[], width=width, height=height, barmode='stack',
        showlegend=False, 
        hoverlabel={'bgcolor': 'black', 'font': {'color': 'white'}, 'namelength': -1}
    )
    layout.update({'xaxis{}'.format(k+(not half)): {'title': names[k]} for k in range2})
    layout.update({'yaxis{}'.format(k+1): {'title': names[k], 'type': 'category', 'autorange': 'reversed'} for k in range1})
    
    # make fig for subplots
    fig = tls.make_subplots(rows=len(range2), cols=len(range1), 
                            shared_xaxes=True, shared_yaxes=True, 
                            vertical_spacing=0.01, horizontal_spacing=0.01, print_grid=False)
    
    for i in range1:
        for j in range2:
            if half and i == j:
                continue
            ct = contingency_table(datasets[j], datasets[i], rownames=[names[j]], colnames=[names[i]])
            n_values = ct.iloc[-1,-1]
            n_labels = len(ct.index) - 1
            if isinstance(colorscale, str):
                clrscl = [c[1] for c in make_colorscale(colorscale, n=n_labels, reverse=kwargs.get('colorscale_reverse', False))]
            
            if ttype == 'percent':
                total_row = ct.iloc[-1] / ct.iloc[-1, -1]
                ct.index = ct.index.astype(str) + ' (' + ct['Total'].astype(str) + ')'
                ct.columns = ct.columns.astype(str) + ' (' + ct.iloc[-1].astype(str) + ')'
                ct = ct / ct.iloc[-1]
                ct.iloc[-1] = total_row.values
                ct = ct * 100
                text = ct.applymap('{:.2f}'.format) + '%'
            else:
                text = ct
            traces = [
                go.Bar(
                    x=ct.iloc[k][:-1], y=ct.columns[:-1], name=ct.index[k], 
                    orientation='h', text=text.iloc[k], hoverinfo='name+text',
                    marker=dict(color=clrscl[k])
                )
            for k in range(n_labels)]

            for trace in traces:
                fig.append_trace(trace, i+1, j+(not half))

    layout.update(kwargs.get('layout', {}))
    fig['layout'].update(layout)
    kwargs.get('save', False) and plty.plot(fig, filename=title+'.html', image_width=width, image_height=height, auto_open=False)
    if kwargs.get('return_fig', False):
        return fig
    else:
        plty.iplot(fig)

def plotly_chi_square_matrix(
        datasets, 
        names:      Optional[Iterable[str]] = None, 
        title:      str = 'Chi-Square Matrix', 
        **kwargs) -> None:
    '''Docstring of `plotly_chi_square_matrix`

    Run and plot chi-square test on every two rows of given datasets,
    with contigency tables calculated on the two rows.

    The chi-squared test is used to determine whether there is a significant 
    difference between the expected frequencies and the observed frequencies
    in one or more categories.

    Args:
        datasets: A list of data to plot.
        names: Data names corresponding to items in data. Optional.
        title: Save the plot with this name.
    '''
    width = kwargs.get('width', 900)
    height = kwargs.get('height', 700)
    # preparation
    datasets = np.array(datasets)
    if kwargs.get('orientation', 'horizontal') == 'vertical':
        datasets = np.array(datasets).T
    ndata = len(datasets)
    if names is not None:
        assert len(names) == ndata, 'Not enough names for data length {}. Given {}.'.format(ndata, len(names))
    else:
        names = ['trace{}'.format(i+1) for i in range(ndata)]
    data = np.c_[names, chi_square_matrix(datasets, names).values]
    data = np.r_[[['']+names], data]
    
    table_layout = kwargs.get('table_layout', {})
    fig = ff.create_table(data, index=True, **table_layout)
    layout = kwargs.get('layout', {})
    layout.update(dict(title=title, width=width, height=height))
    fig['layout'].update(layout)
    kwargs.get('save', False) and plty.plot(fig, filename=title+'.html', image_width=width, image_height=height, auto_open=False)
    if kwargs.get('return_fig', False):
        return fig
    else:
        plty.iplot(fig)

## Machine Learning Visualizers ######################################
def plotly_learning_curve(
        train_sizes, 
        train_means, 
        train_std, 
        val_means, 
        val_stds,
        ylim=None,
        title: str='Learning Curve',
        **kwargs
    ):
    width = kwargs.get('width', 900)
    height = kwargs.get('height', 700)
    layout = go.Layout(
        title=title, width=width, height=height,
        xaxis=dict(title='Training data number'),
        yaxis=dict(title='Score')
    )

    color1, color2 = 'rgb(255,0,0)', 'rgb(0,0,255)'
    trace0 = go.Scatter(
        x=train_sizes, y=train_means+train_std, 
        fill=None, mode='lines', line=dict(width=0), 
        showlegend=False, hoverinfo='skip'
    )
    trace1 = go.Scatter(
        x=train_sizes, y=train_means-train_std, 
        fill='tonexty', mode='none',
        fillcolor='rgba(255, 0, 0, 0.2)', showlegend=False, hoverinfo='skip'
    )
    trace2 = go.Scatter(
        x=train_sizes, y=train_means, fill=None, mode='lines', 
        name='Train score', line=dict(color='rgb(255, 0, 0)')
    )
    trace3 = go.Scatter(
        x=train_sizes, y=val_means+val_stds, 
        fill=None, mode='lines', line=dict(width=0), 
        showlegend=False, hoverinfo='skip'
    )
    trace4 = go.Scatter(
        x=train_sizes, y=val_means-val_stds, 
        fill='tonexty', mode='none',
        fillcolor='rgba(0, 0, 255, 0.2)', showlegend=False, hoverinfo='skip'
    )
    trace5 = go.Scatter(
        x=train_sizes, y=val_means, fill=None, mode='lines', 
        name='Cross-validation score', line=dict(color='rgb(0, 0, 255)')
    )
    if ylim is not None:
        layout.yaxis.range = [*ylim]
    layout.update(kwargs.get('layout', {}))
    fig = go.Figure(data=[trace0, trace1, trace2, trace3, trace4, trace5], layout=layout)
    kwargs.get('save', False) and plty.plot(fig, filename=title+'.html', image_width=width, image_height=height, auto_open=False)
    if kwargs.get('return_fig', False):
        return fig
    else:
        plty.iplot(fig)

## classification
# todo
#  - different shape of boundaries for different models
def plotly_decision_boundary(
        model,
        fx,
        fy,
        y,
        h=0.2,
        bg_colorscale='4,div,RdBu',
        line_colorscale='7,seq,Greys',
        title='Decision Boundaries',
        **kwargs):
    '''Docstring of `plotly_decision_boundary`

    Plot decision boundaries with a model trained with two features.

    The boundary shape is determined by model type.

    Args:
        model: A trained model.
        fx, fy: Two features used to train the given model.
        y: Ground-Truth of every feature pairs.
        h: The feature step for constructing meshgrid.
        bg_colorscale: Background colorscale.
            A list of `[percentage, color]` pairs, or a string
            that can be passed to `make_colorscale`.
        line_colorscale: Probability contour line colors.
            A list of `[percentage, color]` pairs, or a string
            that can be passed to `make_colorscale`.
        title: Save the plot with this name.
    '''
    if model.__class__.__name__ in ['SVC']:
        return plotly_decision_boundary_svm(model, fx, fy, y, h=h, bg_colorscale=bg_colorscale, line_colorscale=line_colorscale, title=title, **kwargs)
    else:
        return plotly_decision_boundary_normal(model, fx, fy, y, h=h, bg_colorscale=bg_colorscale, line_colorscale=line_colorscale, title=title, **kwargs)

def plotly_decision_boundary_normal(
        model,
        fx,
        fy,
        y,
        h=0.2,
        bg_colorscale='4,div,RdBu',
        line_colorscale='7,seq,Greys',
        title='Decision Boundaries',
        **kwargs):
    '''Docstring of `plotly_decision_boundary_normal`

    Plot decision boundaries with a model trained with two features.

    Args:
        model: A model.
        fx, fy: Two features used to train the given model.
        y: Ground-Truth of every feature pairs.
        h: The feature step for constructing meshgrid.
        bg_colorscale: Background colorscale.
            A list of `[percentage, color]` pairs, or a string
            that can be passed to `make_colorscale`.
        line_colorscale: Probability contour line colors.
            A list of `[percentage, color]` pairs, or a string
            that can be passed to `make_colorscale`.
        title: Save the plot with this name.
    '''
    classes, idx = np.unique(y, return_inverse=True)
    n_class = len(classes)
    # train the model
    model.fit(np.c_[fx, fy], idx)
    # background colors
    if isinstance(bg_colorscale, str):
        bg_colorscale = make_colorscale(bg_colorscale, n=n_class)
    # meshgrid
    minx, miny = np.min(fx), np.min(fy)
    maxx, maxy = np.max(fx), np.max(fy)
    dx, dy = np.power(10, np.floor(np.log10(maxx-minx))), np.power(10, np.floor(np.log10(maxy-miny)))
    minx, maxx = minx - dx, maxx + dx
    miny, maxy = miny - dy, maxy + dy
    xrng = np.arange(minx, maxx, h)
    yrng = np.arange(miny, maxy, h)
    xx, yy = np.meshgrid(xrng, yrng)
    xy = np.c_[xx.ravel(), yy.ravel()]
    # layout
    width = kwargs.get('width', 900)
    height = kwargs.get('height', 700)
    layout = go.Layout(
        title=title, width=width, height=height, hovermode='closest',
        xaxis=dict(showgrid=False, range=[minx, maxx], zeroline=False), 
        yaxis=dict(showgrid=False, range=[miny, maxy], zeroline=False)
    )
    layout.update(kwargs.get('layout', {}))
    # class contour; background
    Z = model.predict(xy).reshape(xx.shape).astype(int)
    trace0 = go.Contour(
        z=Z, x=xrng, y=yrng, text=classes, hoverinfo='x+y+text',
        contours=dict(start=0, end=n_class, size=1), line=dict(width=0, smoothing=kwargs.get('contour_smoothing', 0)),
        showscale=False, colorscale=bg_colorscale, opacity=0.6,
    )
    # feature scatters
    trace1 = go.Scattergl(
        x=fx, y=fy, mode='markers', text=y, hoverinfo='x+y+text',
        marker=dict(color=idx, colorscale=bg_colorscale, line=dict(width=1)),
        showlegend=False
    )
    traces = [trace0, trace1]

    # control buttons
    buttons = [dict(
        label = 'Reset',
        method = 'restyle',
        args = [{'visible': [True, True] + [False]*n_class}]
    )]
    # line colors
    if isinstance(line_colorscale, str):
        line_colorscale = make_colorscale(line_colorscale)
    # probrability contours and buttons
    Z2 = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    for i in range(n_class):
        ZZ2 = Z2[:, i].reshape(xx.shape)
        trace = go.Contour(
            z=ZZ2, x=xrng, y=yrng, hoverinfo='x+y+z+name', name=classes[i],
            contours=dict(
                coloring='lines', showlabels=True,
                start=kwargs.get('proba_start', 0), end=kwargs.get('proba_end', 1), size=kwargs.get('proba_step', 0.1)
            ), line=dict(width=2),
            showscale=False, visible=False, colorscale=line_colorscale
        )
        traces.append(trace)
        button = dict(
            label = 'Class {}'.format(classes[i]),
            method = 'restyle',
            args = [{'visible': [True, True] + [False]*i+[True]+[False]*(n_class-i-1)}]
        )
        buttons.append(button)
    buttons.append(dict(
        label = 'All',
        method = 'restyle',
        args = [{'visible': [True, True] + [True]*n_class}]
    ))
    updatemenus = [dict(
        type='buttons',
        buttons=buttons
    )]
    layout.updatemenus = updatemenus

    fig = go.Figure(data=traces, layout=layout)
    kwargs.get('save', False) and plty.plot(fig, filename=title+'.html', image_width=width, image_height=height, auto_open=False)
    if kwargs.get('return_fig', False):
        return fig
    else:
        plty.iplot(fig)

def plotly_decision_boundary_svm(
        model,
        fx,
        fy,
        y,
        h=0.2,
        bg_colorscale='4,div,RdBu',
        line_colorscale=[[0,'rgb(0,0,0)'], [1,'rgb(0,0,0)']],
        title='SVM Decision Boundaries',
        **kwargs):
    '''Docstring of `plotly_decision_boundary_svm`

    Plot decision boundaries with a SVM model trained with two features.

    Args:
        model: A trained model.
        fx, fy: Two features used to train the given model.
        y: Ground-Truth of every feature pairs.
        h: The feature step for constructing meshgrid.
        bg_colorscale: Background colorscale.
            A list of `[percentage, color]` pairs, or a string
            that can be passed to `make_colorscale`.
        line_colorscale: Probability contour line colors.
            A list of `[percentage, color]` pairs, or a string
            that can be passed to `make_colorscale`.
        title: Save the plot with this name.
    '''
    classes, idx = np.unique(y, return_inverse=True)
    n_class = len(classes)
    # train the model
    model.fit(np.c_[fx, fy], idx)
    # background colors
    if isinstance(bg_colorscale, str):
        bg_colorscale = make_colorscale(bg_colorscale, n=n_class)
    # meshgrid
    minx, miny = np.min(fx), np.min(fy)
    maxx, maxy = np.max(fx), np.max(fy)
    dx, dy = np.power(10, np.floor(np.log10(maxx-minx))), np.power(10, np.floor(np.log10(maxy-miny)))
    minx, maxx = minx - dx, maxx + dx
    miny, maxy = miny - dy, maxy + dy
    xrng = np.arange(minx, maxx, h)
    yrng = np.arange(miny, maxy, h)
    xx, yy = np.meshgrid(xrng, yrng)
    xy = np.c_[xx.ravel(), yy.ravel()]
    # layout
    width = kwargs.get('width', 900)
    height = kwargs.get('height', 700)
    layout = go.Layout(
        title=title, width=width, height=height, hovermode='closest',
        xaxis=dict(showgrid=False, range=[minx, maxx], zeroline=False), 
        yaxis=dict(showgrid=False, range=[miny, maxy], zeroline=False)
    )
    layout.update(kwargs.get('layout', {}))
    # class contour; background
    Z = model.predict(xy).reshape(xx.shape).astype(int)
    trace0 = go.Contour(
        z=Z, x=xrng, y=yrng, text=classes, hoverinfo='x+y+text',
        contours=dict(start=0, end=n_class, size=1), line=dict(width=0, smoothing=kwargs.get('contour_smoothing', 0)),
        showscale=False, colorscale=bg_colorscale, opacity=0.6,
    )
    # feature scatters
    trace1 = go.Scattergl(
        x=fx, y=fy, mode='markers', text=y, hoverinfo='x+y+text',
        marker=dict(color=idx, colorscale=bg_colorscale, line=dict(width=1)),
        showlegend=False
    )
    traces = [trace0, trace1]
    
    # control buttons
    buttons = [dict(
        label = 'Reset',
        method = 'restyle',
        args = [{'visible': [True, True] + [False]*3*n_class}]
    )]
    # line colors
    if isinstance(line_colorscale, str):
        line_colorscale = make_colorscale(line_colorscale)
    # train models ovr
    for i in range(n_class):
        c_name = classes[i]
        y_c = np.where(y==c_name, 1, 0)
        m_c = clone(model)
        m_c.fit(np.c_[fx, fy], y_c)
        Z_c = m_c.decision_function(xy).reshape(xx.shape)
        trace_dash = go.Contour(
            z=Z_c, x=xrng, y=yrng, hoverinfo='x+y+z+name', name=c_name,
            contours=dict(
                coloring='lines', showlabels=False,
                start=-1, end=1, size=2
            ), line=dict(width=2, dash='dash'),
            showscale=False, visible=False, colorscale=line_colorscale
        )
        trace_line = go.Contour(
            z=Z_c, x=xrng, y=yrng, hoverinfo='x+y+z+name', name=c_name,
            contours=dict(
                coloring='lines', showlabels=False,
                start=0, end=0, size=0
            ), line=dict(width=2, dash='solid'),
            showscale=False, visible=False, colorscale=line_colorscale
        )
        scatter = go.Scatter(
            x=m_c.support_vectors_[:, 0], y=m_c.support_vectors_[:, 1],
            mode='markers', hoverinfo='x+y+text', text='SupportVector',
            marker=dict(
                symbol='circle-open', color='rgb(0,0,0)', 
                line=dict(width=3)
            ), visible=False, showlegend=False
        )
        traces.append(trace_dash)
        traces.append(trace_line)
        traces.append(scatter)
        button = dict(
            label = 'Class {}'.format(c_name),
            method = 'restyle',
            args = [{'visible': [True, True] + [False]*3*i+[True]*3+[False]*3*(n_class-i-1)}]
        )
        buttons.append(button)
    buttons.append(dict(
        label = 'All',
        method = 'restyle',
        args = [{'visible': [True, True] + [True]*3*n_class}]
    ))
    updatemenus = [dict(
        type='buttons',
        buttons=buttons
    )]
    layout.updatemenus = updatemenus

    fig = go.Figure(data=traces, layout=layout)
    kwargs.get('save', False) and plty.plot(fig, filename=title+'.html', image_width=width, image_height=height, auto_open=False)
    if kwargs.get('return_fig', False):
        return fig
    else:
        plty.iplot(fig)

def tree_boundary(node, x, y, *args, bs=[], line_colorscale=['rgb(0,0,0)']):
    line_styles = ['solid', 'longdashdot', 'longdash', 'dashdot', 'dash', 'dot']
    n_styles = len(line_styles)
    n_colors = len(line_colorscale)
    children_left, children_right, threshold, depth, feature, is_leaves = args
    if not is_leaves[node]:
        child_left = children_left[node]
        child_right = children_right[node]
        t = threshold[node]
        d = depth[node]
        if feature[node] == 0:
            xrng, yrng = [t, t], y
            child_left_x, child_left_y =  [x[0], t], y
            child_right_x, child_right_y =  [t, x[1]], y
        elif feature[node] == 1:
            xrng, yrng = x, [t, t]
            child_left_x, child_left_y =  x, [y[0], t]
            child_right_x, child_right_y = x, [t, y[1]]
        bs.append(go.Scatter(
            x=xrng, y=yrng, mode='lines', name='depth-{}'.format(d),
            line=dict(
                dash=(d<n_styles and line_styles[d] or line_styles[-1]),
                color=(d<n_colors and line_colorscale[d] or line_colorscale[-1])
            ), visible=False
        ))
        bs = tree_boundary(child_left, child_left_x, child_left_y, *args, bs=bs, line_colorscale=line_colorscale)
        bs = tree_boundary(child_right, child_right_x, child_right_y, *args, bs=bs, line_colorscale=line_colorscale)
    return bs

def plotly_decision_boundary_tree(
        model,
        fx,
        fy,
        y,
        h=0.2,
        bg_colorscale='4,div,RdBu',
        line_colorscale=['rgb(0,0,0)'],
        title='Tree Decision Boundaries',
        **kwargs):
    '''Docstring of `plotly_decision_boundary_tree`

    Plot decision boundaries with a Decision Tree model trained with two features.

    Args:
        model: A model.
        fx, fy: Two features used to train the given model.
        y: Ground-Truth of every feature pairs.
        h: The feature step for constructing meshgrid.
        bg_colorscale: Background colorscale.
            A list of `[percentage, color]` pairs, or a string
            that can be passed to `make_colorscale`.
        line_colorscale: Probability contour line colors.
            A list of `[percentage, color]` pairs, or a string
            that can be passed to `make_colorscale`.
        title: Save the plot with this name.
    '''
    classes, idx = np.unique(y, return_inverse=True)
    n_class = len(classes)
    # train the model
    model.fit(np.c_[fx, fy], idx)
    # background colors
    if isinstance(bg_colorscale, str):
        bg_colorscale = make_colorscale(bg_colorscale, n=n_class)
    # meshgrid
    minx, miny = np.min(fx), np.min(fy)
    maxx, maxy = np.max(fx), np.max(fy)
    dx, dy = np.power(10, np.floor(np.log10(maxx-minx))), np.power(10, np.floor(np.log10(maxy-miny)))
    minx, maxx = minx - dx, maxx + dx
    miny, maxy = miny - dy, maxy + dy
    xrng = np.arange(minx, maxx, h)
    yrng = np.arange(miny, maxy, h)
    xx, yy = np.meshgrid(xrng, yrng)
    xy = np.c_[xx.ravel(), yy.ravel()]
    # layout
    width = kwargs.get('width', 900)
    height = kwargs.get('height', 700)
    layout = go.Layout(
        title=title, width=width, height=height, hovermode='closest',
        xaxis=dict(showgrid=False, range=[minx, maxx], zeroline=False), 
        yaxis=dict(showgrid=False, range=[miny, maxy], zeroline=False)
    )
    layout.update(kwargs.get('layout', {}))
    # class contour; background
    Z = model.predict(xy).reshape(xx.shape).astype(int)
    trace0 = go.Contour(
        z=Z, x=xrng, y=yrng, text=classes, hoverinfo='x+y+text',
        contours=dict(start=0, end=n_class, size=1), line=dict(width=0, smoothing=kwargs.get('contour_smoothing', 0)),
        showscale=False, colorscale=bg_colorscale, opacity=0.6,
    )
    # feature scatters
    trace1 = go.Scattergl(
        x=fx, y=fy, mode='markers', text=y, hoverinfo='x+y+text',
        marker=dict(color=idx, colorscale=bg_colorscale, line=dict(width=1)),
        showlegend=False
    )
    traces = [trace0, trace1]

    # generate boundaries
    n_nodes = model.tree_.node_count
    children_left = model.tree_.children_left
    children_right = model.tree_.children_right
    threshold = model.tree_.threshold
    is_branch = np.where(threshold>=0, 1, 0)
    n_boundaries = np.sum(is_branch>0)
    is_left = [1 if i in children_left else 0 for i in range(n_nodes)]
    feature = model.tree_.feature
    depth = np.zeros(n_nodes)
    for i in range(n_nodes):
        if children_left[i] > 0 and children_right[i] > 0:
            depth[children_left[i]] = depth[i] + 1
            depth[children_right[i]] = depth[i] + 1
    tree_info = pd.DataFrame(
        np.c_[is_branch, is_left, feature, threshold, 
              depth, children_left, children_right, 
              [xrng[0]]*n_nodes, [xrng[-1]]*n_nodes, [yrng[0]]*n_nodes, [yrng[-1]]*n_nodes],
        columns=['is_branch', 'is_left', 'feature', 'threshold', 'depth', 'child_left', 'child_right', 'xmin', 'xmax', 'ymin', 'ymax']
    )
    # tree_info[['is_branch', 'is_left', 'feature', 'depth', 'child_left', 'child_right']] = tree_info[['is_branch', 'is_left', 'feature', 'depth', 'child_left', 'child_right']].astype(int)
    for i, row in tree_info.iterrows():
        if row.is_branch:
            axis = row.feature == 0 and 'x' or 'y'
            tree_info.iloc[int(row.child_left)][axis+'max'] = row.threshold
            tree_info.iloc[int(row.child_left)][axis+'min'] = row[axis+'min']
            tree_info.iloc[int(row.child_right)][axis+'max'] = row[axis+'max']
            tree_info.iloc[int(row.child_right)][axis+'min'] = row.threshold
    tree_info = tree_info[tree_info.is_branch > 0]
    # line colors
    if isinstance(line_colorscale, str):
        line_colorscale = [c[1] for c in make_colorscale(line_colorscale, n=n_boundaries)]
    n_colors = len(line_colorscale)
    n_styles = len(line_styles)
    boundaries = [(row.depth, go.Scatter(
        x=(row.feature==0 and [row.threshold]*2 or [row.xmin, row.xmax]),
        y=(row.feature==0 and [row.ymin, row.ymax] or [row.threshold]*2),
        mode='lines', name='depth-{}'.format(int(row.depth)),
        line=dict(
            dash=line_styles[int(row.depth if row.depth<n_styles else -1)],
            color=line_colorscale[int(row.depth if row.depth<n_colors else -1)]
        ), visible=False
    )) for row in tree_info.itertuples()]
    boundaries = [b[1] for b in sorted(boundaries, key=lambda item: item[0])]
    # recursively
    # boundaries = tree_boundary(
    #     0, [xrng[0], xrng[-1]], [yrng[0], yrng[-1]],
    #     children_left, children_right, threshold, depth, feature, 
    #     bs=[], line_colorscale=line_colorscale
    # )

    # control buttons
    d = tree_info['depth'].value_counts().sort_index()
    counts = d.values.astype(int)
    d = d.index.astype(int)

    buttons = [dict(
        label = 'Reset', method = 'restyle',
        args = [{'visible': [True, True] + [False]*n_boundaries}]
    )]
    for i in d:
        buttons.append(dict(
            label = 'depth-{}'.format(i), method = 'restyle',
            args = [{
                'visible': np.concatenate(
                    [[True, True]] + [([True] if j == i else [False]) * counts[j] for j in d]
                )
            }]
        ))
    buttons.append(dict(
        label = 'All',
        method = 'restyle',
        args = [{'visible': [True, True] + [True]*n_boundaries}]
    ))
    updatemenus = [dict(
        type='buttons',
        buttons=buttons
    )]
    layout.updatemenus = updatemenus

    fig = go.Figure(data=traces+boundaries, layout=layout)
    kwargs.get('save', False) and plty.plot(fig, filename=title+'.html', image_width=width, image_height=height, auto_open=False)
    if kwargs.get('return_fig', False):
        return fig
    else:
        plty.iplot(fig)

###############################################################################
## Matplotlib functions
###############################################################################
def plt_heatmap(
        data:       np.array, 
        size:       Tuple[int, int] = (28, 28), 
        title:      object = 'Heatmap', 
        **kwargs) -> None:
    plt.figure(figsize=kwargs.get('figsize', (5,5)))
    image = data.reshape(*size)
    plt.imshow(image, cmap=kwargs.get('colorscale', plt.cm.binary), interpolation=kwargs.get('interpolation', None))
    plt.axis('off')
    plt.title(str(title))
    kwargs.get('save', False) and plt.savefig(title+'.png')

def plt_heatmaps(
        data:   List[np.array], 
        titles: Optional[List[Union[str, int]]] = None, 
        ncols:  int = 4, 
        size:   Tuple[int, int] = (28, 28), 
        title:  str = 'Heatmaps.png', 
        **kwargs) -> None:
    ndata = len(data)
    nrows = int(np.ceil(ndata / ncols))
    data = [image.reshape(*size) for image in data]
    if titles is None:
        titles = ['']*ndata
    # n_empty = nrows * ncols - len(images)   
    # images.append(np.zeros((size[0], size[1] * n_empty)))

    # for i in range(nrows):
    #     for j in range(ncols):
    plt.figure(figsize=kwargs.get('figsize', (5,5)))
    plt.suptitle(title)
    for i in range(ndata):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(data[i], cmap=kwargs.get('colorscale', plt.cm.binary), interpolation=kwargs.get('interpolation', None))
        plt.title(titles[i], fontsize=kwargs.get('title_fontsize', 20))
        plt.axis('off')
    plt.subplots_adjust(
        top=kwargs.get('subplot_top', 0.9),
        bottom=kwargs.get('subplot_bottom', 0.1),
        left=kwargs.get('subplot_left', 0.125),
        right=kwargs.get('subplot_right', 0.9),
        hspace=kwargs.get('subplot_hspace', 0.2),
        wspace=kwargs.get('subplot_wspace', 0.2)
    )
    kwargs.get('save', False) and plt.savefig(title+'.png')

def plt_learning_curve(
        train_sizes, 
        train_means, 
        train_std, 
        val_means, 
        val_stds,
        ylim=None,
        title='Learning Curve'):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training data number')
    plt.ylabel('Score')
    plt.grid()
    plt.fill_between(train_sizes, train_means - train_std, train_means + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, val_means - val_stds, val_means + val_stds, alpha=0.1, color='b')
    plt.plot(train_sizes, train_means, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, val_means, 'o-', color='b', label='Cross-validation score')
    plt.legend(loc='best')
    plt.show()

def plt_decision_boundary(model, features):
    assert len(features) > 0 and len(features) < 3, 'Invalid features. Only 1 or 2 features are supported.'
    # make meshgrid
    features = np.array(features)
    mins, maxs = np.min(features, axis=0), np.max(features, axis=0)
    deltas = np.power(10, np.floor(np.log10(maxs-mins)) - 1)
    mins, maxs = mins- deltas, maxs + deltas
    xx, yy = np.meshgrid(np.arange(mins[0], maxs[0], h), np.arange(mins[1], maxs[1], h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.reshape(xx, yy, Z, cmap=plt.cm.Paired)

###############################################################################
## Other plot functions
###############################################################################

def plot_decision_tree(tree_model, title='Decision Tree', **kwargs):
    export_params = kwargs.get('export_params', {})
    dot = export_graphviz(tree_model, out_file=None, **export_params)
    graph = Source(dot, format='png', engine='dot')
    kwargs.get('save', False) and graph.render(filename=title, cleanup=True)
    return graph









#----------------------------------------------------------#
# deprecated




## Contigency table #####################

# def plotly_df_crosstab_heatmap_matrix(
#         df:         pd.DataFrame, 
#         columns:    List[str], 
#         ttype:      str = 'count', 
#         colorscale: Union[str, list] = 'Greens', 
#         title:      str = 'Contingency Table Matrix',
#         **kwargs) -> None:
#     '''Docstring of `plotly_df_crosstab_heatmap_matrix`

#     Plot contigency tables of every two given columns with plotly heatmap.

#     Args:
#         df: A pandas DataFrame.
#         columns: The column names.
#         ttype: Determines how the contigency table is calculated.
#             'count': The counts of every combination.
#             'colper': The percentage of every combination to the 
#             sum of every rows.
#             Defaults to 'count'.
#         colorscale: The color scale to use.
#         title: Save the plot with this name.
#     '''
#     nrows = ncols = len(columns)
#     fig = tls.make_subplots(rows=nrows, cols=ncols, 
#                             shared_xaxes=True, shared_yaxes=True, 
#                             vertical_spacing=0.01, horizontal_spacing=0.01, print_grid=False)
#     width = kwargs.get('width', 950)
#     height = kwargs.get('height', 750)
#     # layout = go.Layout(title=title, annotations=[], width=width, height=height)
#     # for k in range(nrows):
#     #     layout['xaxis{}'.format(k+1)]['title'] = columns[k]
#     #     layout['yaxis{}'.format(k+1)]['title'] = columns[k]
#     #     layout['xaxis{}'.format(k+1)]['type'] = 'category'
#     #     layout['yaxis{}'.format(k+1)]['type'] = 'category'
#     #     layout['yaxis{}'.format(k+1)]['autorange'] = 'reversed'
#     layout = {'xaxis{}'.format(k+1): {'title': columns[k], 'type': 'category'} for k in range(nrows)}
#     layout.update({'yaxis{}'.format(k+1): {'title': columns[k], 'type': 'category', 'autorange': 'reversed'} for k in range(nrows)})
#     layout.update(dict(
#         title=title, annotations=[], width=width, height=height
#     ))
#     layout = go.Layout(layout)
#     for i in range(nrows):
#         for j in range(ncols):
#             ct = df_contingency_table(df, columns[i], columns[j], ttype=ttype)
            
#             annheat = ff.create_annotated_heatmap(z=ct.values, x=list(ct.columns), y=list(ct.index))
#             trace = annheat['data'][0]
#             trace['colorscale'] = colorscale

#             annotations = annheat['layout']['annotations']
#             for ann in annotations:
#                 ann['xref'] = 'x{}'.format(j+1)
#                 ann['yref'] = 'y{}'.format(i+1)
#                 ann['font']['color'] = float(ann['text']) / df.shape[0] > 0.5 and 'rgb(255,255,255)' or 'rgb(0,0,0)'
#                 if ttype == 'colper': ann['text'] = ann['text'] + '%'
#             layout['annotations'] = list(layout['annotations']).extend(annotations)
            
#             fig.append_trace(trace, i+1, j+1)    
            
#     fig['layout'].update(layout)
#     kwargs.get('save', False) and plty.plot(fig, filename=title+'.html', image_width=width, image_height=height, auto_open=False)
#     plty.iplot(fig)

# def plotly_df_crosstab_stacked(
#         df:     pd.DataFrame, 
#         col1:   str, 
#         col2:   str, 
#         title:  str = 'crosstab_stacked_bar',
#         **kwargs) -> None:
#     '''Docstring of `plotly_df_crosstab_stacked`

#     Plot stacked bar of two given columns' contigency table with plotly heatmap.

#     Args:
#         df: A pandas DataFrame.
#         col1: Index of the contigency table.
#         col2: Column of the contigency table.
#         title: Save the plot with this name.
#     '''
#     ct = df_contingency_table(df, col1, col2)
#     width = kwargs.get('width', 900)
#     height = kwargs.get('height', 700)
#     layout = go.Layout(
#         barmode = 'stack',
#         title = '{}-{}'.format(ct.index.name, ct.columns.name),
#         yaxis = dict(title=ct.columns.name),
#         annotations = [
#             dict(
#                 x=1.12,
#                 y=1.05,
#                 text='Pclass',
#                 showarrow=False,
#                 xref="paper",
#                 yref="paper",
#             )
#         ],
#         width=width,
#         height=height
#     )
#     ct.index = ct.index.astype(str) + ' <br>(n=' + ct['Total'].astype(str) + ')'
#     ct.columns = ct.columns.astype(str) + ' <br>(n=' + ct.iloc[-1].astype(str) + ')'
#     ct = (ct / ct.iloc[-1] * 100).round().astype(int)
#     data = [go.Bar(x=ct.iloc[i][:-1], y=ct.columns[:-1], name=ct.index[i], orientation='h') for i in range(ct.index.shape[0]-1)]
    
#     fig = go.Figure(data=data, layout=layout)
#     kwargs.get('save', False) and plty.plot(fig, filename=title+'.html', image_width=width, image_height=height, auto_open=False)
#     plty.iplot(fig)

# def plotly_df_crosstab_stacked_matrix(
#         df:         pd.DataFrame, 
#         columns:    List[str], 
#         colorscale: Union[str, list] = 'Greens', 
#         title:      str = 'Stacked Bar Matrix',
#         **kwargs) -> None:
#     '''Docstring of `plotly_df_crosstab_stacked_matrix`

#     Plot stacked bars of every two given columns' contigency table with plotly heatmap.

#     Args:
#         df: A pandas DataFrame.
#         columns: The column names.
#         colorscale: The color scale to use.
#         title: Save the plot with this name.
#     '''
#     nrows = ncols = len(columns)
#     fig = tls.make_subplots(rows=nrows, cols=ncols, 
#                             shared_xaxes=True, shared_yaxes=True, 
#                             vertical_spacing=0.01, horizontal_spacing=0.01, print_grid=False)
#     width = kwargs.get('width', 950)
#     height = kwargs.get('height', 750)
#     # layout = go.Layout(title=title, annotations=[], 
#     #                     width= width, height=height, barmode='stack',
#     #                     showlegend=False, hoverlabel={'bgcolor': 'black', 'font': {'color': 'white'}, 'namelength': -1})
#     # for k in range(nrows):
#     #     layout['xaxis{}'.format(k+1)]['title'] = columns[k]
#     #     layout['yaxis{}'.format(k+1)]['title'] = columns[k]
#     #     #layout['xaxis{}'.format(k+1)]['type'] = 'category'
#     #     layout['yaxis{}'.format(k+1)]['type'] = 'category'
#     #     layout['yaxis{}'.format(k+1)]['autorange'] = 'reversed'
#     layout = {'xaxis{}'.format(k+1): {'title': columns[k]} for k in range(nrows)}
#     layout.update({'yaxis{}'.format(k+1): {'title': columns[k], 'type': 'category', 'autorange': 'reversed'} for k in range(nrows)})
#     layout.update(dict(
#         title=title, annotations=[], 
#         width= width, height=height, barmode='stack',
#         showlegend=False, hoverlabel={'bgcolor': 'black', 'font': {'color': 'white'}, 'namelength': -1}
#     ))
#     layout = go.Layout(layout)
#     for i in range(nrows):
#         for j in range(ncols):
#             ct = df_contingency_table(df, columns[j], columns[i])
#             ct.index = ct.index.astype(str) + ' <br>(n=' + ct['Total'].astype(str) + ')'
#             ct.columns = ct.columns.astype(str) + ' <br>(n=' + ct.iloc[-1].astype(str) + ')'
#             ct = (ct / ct.iloc[-1] * 100).round().astype(int)
#             data = [go.Bar(x=ct.iloc[k][:-1], y=ct.columns[:-1], name=ct.index[k], orientation='h') for k in range(ct.index.shape[0]-1)]
            
#             for trace in data:
#                 fig.append_trace(trace, i+1, j+1)
    
#     fig['layout'].update(layout)
#     kwargs.get('save', False) and plty.plot(fig, filename=title+'.html', image_width=width, image_height=height, auto_open=False)
#     plty.iplot(fig)

# def plotly_qq_plots(
#         data:   list, 
#         names:  list = [], 
#         ncols:  Optional[int] = None,
#         title:  str = 'QQ plots',
#         **kwargs) -> None:
#     '''Docstring of `plotly_describes`

#     Plot QQ-plots of all the given data with plotly.

#     Args:
#         data: A list of numerical data.
#         names: A list of names corresponding to data rows.
#         ncols: Number of subplots of every row.
#             If `None`, it's determined with number of data.
#         title: Save the plot with this name.
#     '''
#     ndata = len(data)
#     names = names or ['']*ndata
#     if ncols is None:
#         nrows = int(np.floor(np.power(ndata, 0.5)))
#         ncols = int(np.ceil(ndata / nrows))
#     else:
#         nrows = int(np.ceil(ndata / ncols))
#     fig = tls.make_subplots(rows=nrows, cols=ncols, subplot_titles=names, 
#                             vertical_spacing=0.1, horizontal_spacing=0.1, print_grid=False)
#     for i in range(nrows):
#         for j in range(ncols):
#             try:
#                 p = stats.probplot(data[ncols * i + j])
#             except:
#                 break
#             fig.append_trace(go.Scattergl(x=p[0][0], y=p[0][1], mode='markers'), i+1, j+1)
#             fig.append_trace(go.Scattergl(x=p[0][0], y=p[0][0]*p[1][0]+p[1][1]), i+1, j+1)
    
#     width = kwargs.get('width', 900)
#     height = kwargs.get('height', 700)
#     layout = go.Layout(title=title, showlegend=False, width=width, height=height)
#     fig['layout'].update(layout)
#     kwargs.get('save', False) and plty.plot(fig, filename=title+'.html', image_width=width, image_height=height, auto_open=False)
#     plty.iplot(fig)
