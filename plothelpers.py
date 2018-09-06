from basichelpers import *
import colorlover as cl
import plotly.offline as plty
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

###############################################################################
## Plotly functions
###############################################################################
## basics
def make_colorscale(cname: str, n: int, cnum: str = '3', ctype: str = 'seq') -> list:
    n = n // 10 * 10
    return [[i/(n-1), c] for i, c in  enumerate(cl.to_rgb(cl.interp(cl.scales[cnum][ctype][cname], n)))]

#######################
def plotly_df_categorical_bar(df: pd.DataFrame, columns: List[str], ncols: int = 4, save: bool = False, filename: str = 'bar_charts', **kwargs) -> None:
    '''Docstring of `plotly_df_categorical_bar`

    Plot categorical columns of given DataFrame with plotly.

    Args:
        df: A pandas DataFrame.
        columns: The column names.
        ncols: Number of subplots of every row.
        save: Whether to save the plot or not.
        filename: Save the plot with this name.
    '''
    nrows = int(np.ceil(len(columns) / ncols))
    fig = tls.make_subplots(rows=nrows, cols=ncols, subplot_titles=columns, print_grid=False)
    for i in range(nrows):
        for j in range(ncols):
            try:
                s = df[columns[ncols * i + j]].value_counts()
            except:
                break
            trace = go.Bar(x=s.index, y=s.values, name=s.name)
            fig.append_trace(trace, i+1, j+1)
    save and plty.plot(fig, filename=filename+'.html', auto_open=False)
    plty.iplot(fig)

def plotly_df_numerical_hist(df: pd.DataFrame, columns: List[str], ncols: int = 4, save: bool = False, filename: str = 'histograms', **kwargs) -> None:
    '''Docstring of `plotly_df_numerical_hist`

    Plot numerical columns of given DataFrame with plotly.

    Args:
        df: A pandas DataFrame.
        columns: The column names.
        ncols: Number of subplots of every row.
        save: Whether to save the plot or not.
        filename: Save the plot with this name.
    '''
    nrows = int(np.ceil(len(columns) / ncols))
    fig = tls.make_subplots(rows=nrows, cols=ncols, subplot_titles=columns, print_grid=False)
    
    for i in range(nrows):
        for j in range(ncols):
            try:
                s = df[columns[ncols * i + j]]
            except:
                break
            trace = go.Histogram(x=s, xbins=dict(start=s.min(), end=s.max(), size=kwargs.get('size', None)), name=s.name)
            fig.append_trace(trace, i+1, j+1)
    save and plty.plot(fig, filename=filename+'.html', auto_open=False)
    plty.iplot(fig)

def plotly_df_grouped_hist(df: pd.DataFrame, col1: str, col2: str, ncols: int = 4, normdist: bool = False, save: bool = False, filename: str = 'histograms', **kwargs) -> None:
    '''Docstring of `plotly_df_grouped_hist`

    Plot histograms of given column grouped on the unique value of another column with plotly.

    Args:
        df: A pandas DataFrame.
        col1: Name of the column that the unique values of which
            will be used for grouping.
        col2: Name of the column that will be grouped.
        ncols: Number of subplots of every row.
        normdist: Whether plot the normal distribution with mean 
            and std of given data.
        save: Whether to save the plot or not.
        filename: Save the plot with this name.
    '''
    vals, data = grouped_col(df, col1, col2)
    nrows = int(np.ceil(len(vals) / ncols))
    fig = tls.make_subplots(rows=nrows, cols=ncols, subplot_titles=[str(v) for v in vals],
                            shared_yaxes=True, print_grid=False)
    layout = go.Layout({'title': 'Histogram of {} - grouped by {}'.format(col2, col1), 'showlegend': False})
    for k in range(nrows):
        layout['yaxis{}'.format(k+1)]['title'] = 'Count'
    
    for i in range(nrows):
        for j in range(ncols):
            try:
                s = data[ncols * i + j]
            except:
                break
            if normdist:
                tempfig = ff.create_distplot([s], [s.name], curve_type='normal', histnorm='')
                for trace in tempfig['data']:
                    trace['xaxis'] = 'x{}'.format(i+1)
                    trace['yaxis'] = 'y{}'.format(j+1)
                    if trace['type'] == 'scatter':
                        trace['y'] = trace['y'] * s.count()
                    fig.append_trace(trace, i+1, j+1)
            else:
                trace = go.Histogram(x=s, xbins=dict(start=s.min(), end=s.max(), size=kwargs.get('size', None)), name=s.name)
                fig.append_trace(trace, i+1, j+1)
    fig['layout'].update(layout)
    save and plty.plot(fig, filename=filename+'.html', auto_open=False)
    plty.iplot(fig)
    
#######################
def plotly_df_crosstab_heatmap(df: pd.DataFrame, col1: str, col2: str, ttype: str = 'count', title: bool = False, axes_title: bool = False, save: bool = False, filename: str = 'crosstab_heatmap') -> None:
    '''Docstring of `plotly_df_crosstab_heatmap`

    Plot contigency table of two given columns with plotly heatmap.

    Args:
        df: A pandas DataFrame.
        col1: Index of the contigency table.
        col2: Column of the contigency table.
        ttype: Determines how the contigency table is calculated.
            'count': The counts of every combination.
            'colper': The percentage of every combination to the 
            sum of every rows.
            Defaults to 'count'.
        title: Whether to show the plot title or not.
        axes_title: Whether to show the axis' title or not.
        save: Whether to save the plot or not.
        filename: Save the plot with this name.
    '''
    ct = df_contingency_table(df, col1, col2, ttype=ttype)
    fig = ff.create_annotated_heatmap(z=ct.values, x=list(ct.columns), y=list(ct.index))
    fig['layout']['title'] = title and '{}-{}'.format(ct.index.name, ct.columns.name)
    fig['layout']['xaxis']['title'] = axes_title and ct.columns.name
    fig['layout']['yaxis']['title'] = axes_title and ct.index.name
    fig['layout']['xaxis']['side'] = 'bottom'
    save and plty.plot(fig, filename=filename+'.html', auto_open=False)
    plty.iplot(fig)

def plotly_df_crosstab_heatmap_matrix(df: pd.DataFrame, columns: List[str], ttype: str = 'count', colorscale: Union[str, list] = 'Greens', width: int = 950, height: int = 750, save: bool = False, filename: str = 'crosstab_heatmap_matrix') -> None:
    '''Docstring of `plotly_df_crosstab_heatmap_matrix`

    Plot contigency tables of every two given columns with plotly heatmap.

    Args:
        df: A pandas DataFrame.
        columns: The column names.
        ttype: Determines how the contigency table is calculated.
            'count': The counts of every combination.
            'colper': The percentage of every combination to the 
            sum of every rows.
            Defaults to 'count'.
        colorscale: The color scale to use.
        width: Plot width.
        height: Plot height.
        save: Whether to save the plot or not.
        filename: Save the plot with this name.
    '''
    nrows = ncols = len(columns)
    fig = tls.make_subplots(rows=nrows, cols=ncols, 
                            shared_xaxes=True, shared_yaxes=True, 
                            vertical_spacing=0.01, horizontal_spacing=0.01, print_grid=False)
    layout = go.Layout({'title': 'Contingency Table Matrix', 'annotations': [], 'width': width, 'height': height})
    for k in range(nrows):
        layout['xaxis{}'.format(k+1)]['title'] = columns[k]
        layout['yaxis{}'.format(k+1)]['title'] = columns[k]
        layout['xaxis{}'.format(k+1)]['type'] = 'category'
        layout['yaxis{}'.format(k+1)]['type'] = 'category'
        layout['yaxis{}'.format(k+1)]['autorange'] = 'reversed'
        
    for i in range(nrows):
        for j in range(ncols):
            ct = df_contingency_table(df, columns[i], columns[j], ttype=ttype)
            
            annheat = ff.create_annotated_heatmap(z=ct.values, x=list(ct.columns), y=list(ct.index))
            trace = annheat['data'][0]
            trace['colorscale'] = colorscale

            annotations = annheat['layout']['annotations']
            for ann in annotations:
                ann['xref'] = 'x{}'.format(j+1)
                ann['yref'] = 'y{}'.format(i+1)
                ann['font']['color'] = float(ann['text']) / df.shape[0] > 0.5 and 'rgb(255,255,255)' or 'rgb(0,0,0)'
                if ttype == 'colper': ann['text'] = ann['text'] + '%'
            layout['annotations'].extend(annotations)
            
            fig.append_trace(trace, i+1, j+1)    
            
    fig['layout'].update(layout)
    save and plty.plot(fig, filename=filename+'.html', image_width=width, image_height=height, auto_open=False)
    plty.iplot(fig)

def plotly_df_crosstab_stacked(df: pd.DataFrame, col1: str, col2: str, save: bool = False, filename: str = 'crosstab_stacked_bar') -> None:
    '''Docstring of `plotly_df_crosstab_stacked`

    Plot stacked bar of two given columns' contigency table with plotly heatmap.

    Args:
        df: A pandas DataFrame.
        col1: Index of the contigency table.
        col2: Column of the contigency table.
        save: Whether to save the plot or not.
        filename: Save the plot with this name.
    '''
    ct = df_contingency_table(df, col1, col2)
    layout = go.Layout(
        barmode = 'stack',
        title = '{}-{}'.format(ct.index.name, ct.columns.name),
        yaxis = dict(title=ct.columns.name),
        annotations = [
            dict(
                x=1.12,
                y=1.05,
                text='Pclass',
                showarrow=False,
                xref="paper",
                yref="paper",
            )
        ]
    )
    ct.index = ct.index.astype(str) + ' <br>(n=' + ct['Total'].astype(str) + ')'
    ct.columns = ct.columns.astype(str) + ' <br>(n=' + ct.iloc[-1].astype(str) + ')'
    ct = (ct / ct.iloc[-1] * 100).round().astype(int)
    data = [go.Bar(x=ct.iloc[i][:-1], y=ct.columns[:-1], name=ct.index[i], orientation='h') for i in range(ct.index.shape[0]-1)]
    
    fig = go.Figure(data=data, layout=layout)
    save and plty.plot(fig, filename=filename+'html', auto_open=False)
    plty.iplot(fig)

def plotly_df_crosstab_stacked_matrix(df: pd.DataFrame, columns: List[str], colorscale: Union[str, list] = 'Greens', width: int = 950, height: int = 750, save: bool = False, filename: str = 'crosstab_stacked_matrix') -> None:
    '''Docstring of `plotly_df_crosstab_stacked_matrix`

    Plot stacked bars of every two given columns' contigency table with plotly heatmap.

    Args:
        df: A pandas DataFrame.
        columns: The column names.
        colorscale: The color scale to use.
        width: Plot width.
        height: Plot height.
        save: Whether to save the plot or not.
        filename: Save the plot with this name.
    '''
    nrows = ncols = len(columns)
    fig = tls.make_subplots(rows=nrows, cols=ncols, 
                            shared_xaxes=True, shared_yaxes=True, 
                            vertical_spacing=0.01, horizontal_spacing=0.01, print_grid=False)
    layout = go.Layout({'title': 'Stacked Bar Matrix', 'annotations': [], 
                        'width': width, 'height': height, 'barmode': 'stack',
                        'showlegend': False, 'hoverlabel': {'bgcolor': 'black', 'font': {'color': 'white'}, 'namelength': -1}})
    for k in range(nrows):
        layout['xaxis{}'.format(k+1)]['title'] = columns[k]
        layout['yaxis{}'.format(k+1)]['title'] = columns[k]
        #layout['xaxis{}'.format(k+1)]['type'] = 'category'
        layout['yaxis{}'.format(k+1)]['type'] = 'category'
        layout['yaxis{}'.format(k+1)]['autorange'] = 'reversed'
    for i in range(nrows):
        for j in range(ncols):
            ct = df_contingency_table(df, columns[j], columns[i])
            ct.index = ct.index.astype(str) + ' <br>(n=' + ct['Total'].astype(str) + ')'
            ct.columns = ct.columns.astype(str) + ' <br>(n=' + ct.iloc[-1].astype(str) + ')'
            ct = (ct / ct.iloc[-1] * 100).round().astype(int)
            data = [go.Bar(x=ct.iloc[k][:-1], y=ct.columns[:-1], name=ct.index[k], orientation='h') for k in range(ct.index.shape[0]-1)]
            
            for trace in data:
                fig.append_trace(trace, i+1, j+1)
    
    fig['layout'].update(layout)
    save and plty.plot(fig, filename=filename+'.html', image_width=width, image_height=height, auto_open=False)
    plty.iplot(fig)
    
#######################
def plotly_df_box(df: pd.DataFrame, col1: str, col2: str, save: bool = False, filename: str = 'Box Plot') -> None:
    '''Docstring of `plotly_df_crosstab_stacked`

    Plot box-plot of one column grouped by the unique value of another column with plotly.

    Args:
        df: A pandas DataFrame.
        col1: Name of the column that the unique values of which
            will be used for grouping.
        col2: Name of the column that will be grouped.
        save: Whether to save the plot or not.
        filename: Save the plot with this name.
    '''
    # df = df[[col1, col2]].dropna()
    # cols = df[col1].unique()
    # traces = [go.Box(y=df[df[col1] == col][col2], boxmean='sd', name=col) for col in cols]
    vals, data = grouped_col(df, col1, col2)
    traces = [go.Box(y=d, boxmean='sd', name=v) for v,d in zip(vals, data)]

    layout = go.Layout(
        title='{} boxes grouped by {}'.format(col2, col1),
        yaxis=dict(title=col2),
        xaxis=dict(title=col1)
    )
    fig = go.Figure(data=traces, layout=layout)
    save and plty.plot(fig, filename=filename+'.html', auto_open=False)
    plty.iplot(fig)
    
def plotly_df_chi_square_matrix(df: pd.DataFrame, columns: List[str], cell_height: int = 45, width: int = 900, height: int = 700, save: bool = False, filename: str = 'Chi-Square Matrix') -> None:
    '''Docstring of `plotly_df_chi_square_matrix`

    Run chi-square test on every two columns of given pandas DataFrame,
    with contigency tables calculated on the two columns.
    Then plot the results as a matrix.

    Args:
        df: A pandas DataFrame.
        columns: A list contains columns to run chi-square test with.
        save: Whether to save the plot or not.
        filename: Save the plot with this name.
    '''
    data = np.c_[columns, df_chi_square_matrix(df, columns).values]
    data = np.r_[[['']+columns], data]
    fig = ff.create_table(data, height_constant=cell_height, index=True)
    fig.layout.width = width
    fig.layout.height = height
    save and plty.plot(fig, filename=filename+'.html', auto_open=False)
    plty.iplot(fig)

def plotly_describes(data: list, names: list = [], width: int = 900, height: int = 700, save: bool = False, filename: str = 'Descriptive Statistics'):
    '''Docstring of `plotly_describes`

    Plot a table of descriptive statistics of given data with plotly.

    Args:
        data: A list of numerical data.
        names: A list contains names corresponding to data.
        save: Whether to save the plot or not.
        filename: Save the plot with this name.
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
        describes[1:, i+1] = [int(v*10000)/10000 for v in des[2:, 1]]
    fig = ff.create_table(describes, index=True)
    fig.layout.width = width
    fig.layout.height = height
    save and plty.plot(fig, filename=filename+'.html', auto_open=False)
    plty.iplot(fig)

def plotly_qq_plots(data: list, names: list = [], ncols: int = 4, width: int = 900, height: int = 700, save: bool = False, filename: str = 'QQ plots'):
    '''Docstring of `plotly_describes`

    Plot QQ-plots of given data with plotly.

    Args:
        data: A list of numerical data.
        names: A list contains names corresponding to data.
        ncols: Number of subplots of every row.
        save: Whether to save the plot or not.
        filename: Save the plot with this name.
    '''
    ndata = len(data)
    names = names or ['']*ndata
    nrows = int(np.ceil(ndata / ncols))
    fig = tls.make_subplots(rows=nrows, cols=ncols, subplot_titles=names, 
                            vertical_spacing=0.1, horizontal_spacing=0.1, print_grid=False)
    layout = go.Layout({'title': 'QQ plots', 'showlegend': False})
    for i in range(nrows):
        for j in range(ncols):
            p = stats.probplot(data[ncols * i + j])
            fig.append_trace(go.Scatter(x=p[0][0], y=p[0][1], mode='markers'), i+1, j+1)
            fig.append_trace(go.Scatter(x=p[0][0], y=p[0][0]*p[1][0]+p[1][1]), i+1, j+1)
    
    fig['layout'].update(layout)
    save and plty.plot(fig, filename=filename+'.html', image_width=width, image_height=height, auto_open=False)
    plty.iplot(fig)

#######################
def plotly_df_2d_clusters(df: pd.DataFrame, x: str, y: str, l: str, title: Optional[str] = '2D Clusters', colors: Union[str, list] = '12,qual,Paired', **kwargs):
    '''Docstring of `plotly_df_2d_clusters`

    Plot scatter plots of two columns grouped by the unique values
    of a third column with plotly.

    Args:
        df: A pandas DataFrame.
        x: The column name of x values.
        y: The column name of y values.
        l: The column name to group x and y on which.
        title: Title of the plot.
        colors: Colors of every clusters. Accept a list of color strings
        with same length as the number of clusters. Or a string that will
        be passed to `make_colorscale` to make colors for every clusters.
    '''
    # group by unique values
    dfs = dict(tuple(df[[x, y, l]].groupby(l)))
    labels = np.unique(df[l])
    n_labels = len(labels)
    # check for colors
    if isinstance(colors, str):
        c1, c2, c3 = colors.split(',')
        colors =  [c[1] for c in make_colorscale(c3, n_labels, c1, c2)]
    else:
        assert len(colors) == n_labels, 'Invalid colors. {} colors is needed.'.format(n_labels)
    # add data points cluster by cluster
    traces = []
    opacitys = [kwargs.get('inactive_opacity', 0.05),] * n_labels
    line_width = [kwargs.get('inactive_line_width', 0.1),] * n_labels
    # buttons for selecting cluster to highlight
    buttons = [dict(
        label = 'Reset',
        method = 'restyle',
        args = [{
            'marker.opacity': [kwargs.get('active_opacity', 1),] * n_labels,
            'marker.line.width': [kwargs.get('active_line_width', 1),]*n_labels
        }]
    )]
    for i, (label, color) in enumerate(zip(labels, colors)):
        data = dfs[label]
        trace = go.Scattergl(
            x=data[x], y=data[y], mode='markers', name=label, 
            marker=dict(color=color, opacity=1, line=dict(width=1)))
        traces.append(trace)
        button = dict(
            label = 'Cluster {}'.format(label),
            method = 'restyle',
            args = [{
                'marker.opacity': opacitys[:i] + [kwargs.get('active_opacity', 1),] + opacitys[i+1:],
                'marker.line.width': line_width[:i] + [kwargs.get('active_line_width', 1),] + line_width[i+1:],
            }]
        )
        buttons.append(button)
    updatemenus = [
        dict(
            type='buttons',
            buttons=buttons
        )
    ]
    layout = go.Layout(title=title, updatemenus=updatemenus)
    fig = go.Figure(data=traces, layout=layout)
    kwargs.get('save', False) and plty.plot(fig, filename=title+'.html', image_width=kwargs.get('width', 900), image_height=kwargs.get('height', 700), auto_open=False)
    plty.iplot(fig)

def plotly_2d_clusters(data: np.array, title: Optional[str] = '2D Clusters', colors: Union[str, list] = '12,qual,Paired', **kwargs):
    '''Docstring of `plotly_2d_clusters`

    Plot scatter plots of first two columns grouped by the unique values
    of the third column with plotly.

    Args:
        data: A numpy array.
        title: Title of the plot.
        colors: Colors of every clusters. Accept a list of color strings
        with same length as the number of clusters. Or a string that will
        be passed to `make_colorscale` to make colors for every clusters.
    '''
    # group by unique values
    labels = np.unique(data[:,2])
    data = {k: data[data[:,2]==k] for k in labels}
    n_labels = len(labels)
    # check for colors
    if isinstance(colors, str):
        c1, c2, c3 = colors.split(',')
        colors =  [c[1] for c in make_colorscale(c3, n_labels, c1, c2)]
    else:
        assert len(colors) == n_labels, 'Invalid colors. {} colors is needed.'.format(n_labels)
    opacitys = [kwargs.get('inactive_opacity', 0.05),] * n_labels
    line_width = [kwargs.get('inactive_line_width', 0.1),] * n_labels
    # buttons for selecting cluster to highlight
    buttons = [dict(
        label = 'Reset',
        method = 'restyle',
        args = [{
            'marker.opacity': [kwargs.get('active_opacity', 1),] * n_labels,
            'marker.line.width': [kwargs.get('active_line_width', 1),]*n_labels
        }]
    )]
    # add data points cluster by cluster
    traces = []
    for i, (label, color) in enumerate(zip(labels, colors)):
        d = data[label]
        trace = go.Scattergl(
            x=d[:,0], y=d[:,1], mode='markers', name=label, 
            marker=dict(color=color, opacity=1, line=dict(width=1)))
        traces.append(trace)
        button = dict(
            label = 'Cluster {}'.format(label),
            method = 'restyle',
            args = [{
                'marker.opacity': opacitys[:i] + [kwargs.get('active_opacity', 1),] + opacitys[i+1:],
                'marker.line.width': line_width[:i] + [kwargs.get('active_line_width', 1),] + line_width[i+1:],
            }]
        )
        buttons.append(button)
    updatemenus = [
        dict(
            type='buttons',
            buttons=buttons
        )
    ]
    layout = go.Layout(title=title, updatemenus=updatemenus)
    fig = go.Figure(data=traces, layout=layout)
    kwargs.get('save', False) and plty.plot(fig, filename=title+'.html', image_width=kwargs.get('width', 900), image_height=kwargs.get('height', 700), auto_open=False)
    plty.iplot(fig)

###############################################################################
## Matplotlib functions
###############################################################################

#######################
def plt_heatmap(data: np.array, title: Optional[Union[str, int]] = None, size: Tuple[int, int] = (28, 28), save: bool = False, filename: str = 'Heatmap.png', **kwargs) -> None:
    plt.figure(figsize=kwargs.get('figsize', (5,5)))
    image = data.reshape(*size)
    plt.imshow(image, cmap=kwargs.get('colorscale', plt.cm.binary), interpolation=kwargs.get('interpolation', None))
    plt.axis('off')
    title and plt.title(title)
    save and plt.savefig(filename)

def plt_heatmaps(data: List[np.array], titles: Optional[List[Union[str, int]]], ncols: int = 4, size: Tuple[int, int] = (28, 28), save: bool = False, filename: str = 'Heatmaps.png', **kwargs) -> None:
    ndata = len(data)
    nrows = int(np.ceil(ndata / ncols))
    data = [image.reshape(*size) for image in data]
    if not np.array(titles).size:
        titles = ['']*ndata
    # n_empty = nrows * ncols - len(images)   
    # images.append(np.zeros((size[0], size[1] * n_empty)))

    # for i in range(nrows):
    #     for j in range(ncols):
    plt.figure(figsize=kwargs.get('figsize', (5,5)))
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
    save and plt.savefig(filename)