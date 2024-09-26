import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go


def Plot3D(values, options=None, title=None):
    """
    Plot a 3D dataset based on various options and return the figure for inline display in Jupyter Notebook.

    Args:
        values (np.array): 3D numpy array
        options (Dict):
            - filter: Drop values smaller than mean + num_stddev * std. Takes a float value.
            - sample: Randomly remove specified percentage of data. Takes a float value between 0 and 1.
            - nozero: Drop 0 values. Takes True.
            - fix_range: Fix the color range. Takes a list of two float values.
    """

    shape = values.shape
    indices = np.indices((shape[0], shape[1], shape[2]))
    x = indices[0].flatten()
    y = indices[1].flatten()
    z = indices[2].flatten()
    values_flat = values.flatten()

    df = pd.DataFrame({
        'x': x,
        'y': y,
        'z': z,
        'values': values_flat
    })

    df.dropna(inplace=True)  # drop nan values

    if options is None:
        options = {}

    if "filter" in options:
        num_stddevs = options["filter"]
        df = df[df['values'] > df['values'].mean() + num_stddevs * df['values'].std()]
    if "sample" in options:
        sample_rate = options["sample"]
        df = df.sample(frac=sample_rate)
    if "nozero" in options:
        df = df[df['values'] != 0]
    if "fix_range" in options:
        range_color = options["fix_range"]
    else:
        range_color = [df['values'].min(), df['values'].max()]

    fig = px.scatter_3d(df, x='x', y='y', z='z', color='values', opacity=0.1, range_color=range_color)
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(height=500)

    if title is not None:
        fig.update_layout(title=title)

    # Set fixed axes limits
    fig.update_layout(scene=dict(
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[0, 88]),
        zaxis=dict(range=[0, 32])
    ))

    fig.update_coloraxes(colorbar_title='magnitude')

    return fig  # Return the figure for inline display


def Plot2D(values, title=None, axis=0, depth=0, mean_over_z=True):
    """
    Plot 2D slices or the mean over the z-axis of the 3D array at specified depth and return the figure for inline display in Jupyter Notebook.
    axis = 0: around x
    axis = 1: around y
    axis = 2: around z
    mean_over_z: If True, compute and plot the mean values over the z-axis.
    """

    if mean_over_z:
        # Compute the mean over the z-axis (axis=2)
        slice_ = np.mean(values, axis=2)
    else:
        if axis == 0:
            slice_ = values[depth, :, :]
        elif axis == 1:
            slice_ = values[:, depth, :]
        else:  # axis == 2
            slice_ = values[:, :, depth]

    fig = px.imshow(slice_)

    if title is not None:
        fig.update_layout(title=title)

    return fig  # Return the figure for inline display


def PlotLoss(history):
    """
    Plot the loss and accuracy of the training and validation sets.
    """
    # Create a copy of the relevant parts of the history dictionary
    train_r2 = [max(0, x) for x in history['train_r2']]
    validation_r2 = [max(0, x) for x in history['validation_r2']]
    train_loss = history['train_loss'][:]
    validation_loss = history['validation_loss'][:]
    current_epochs = history['current_epochs']

    # Set the first 10 elements of train_loss and validation_loss to zero
    train_loss[:10] = [0] * 10
    validation_loss[:10] = [0] * 10

    # Create subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Accuracy', 'Loss'))

    # Add traces for loss
    fig.add_trace(go.Scatter(x=np.arange(current_epochs), y=train_loss, mode='lines', name='Train loss'), row=1, col=2)
    fig.add_trace(go.Scatter(x=np.arange(current_epochs), y=validation_loss, mode='lines', name='Validation loss'), row=1, col=2)

    # Add traces for R2
    fig.add_trace(go.Scatter(x=np.arange(current_epochs), y=train_r2, mode='lines', name='Train R2'), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(current_epochs), y=validation_r2, mode='lines', name='Validation R2'), row=1, col=1)

    # Update axes titles
    fig.update_xaxes(title_text='Epochs', row=1, col=1)
    fig.update_yaxes(title_text='R2', row=1, col=1, range=[0, 1])
    fig.update_xaxes(title_text='Epochs', row=1, col=2)
    fig.update_yaxes(title_text='MSE', row=1, col=2, range=[0, None])

    # Add title
    fig.update_layout(title_text="Convergence", showlegend=True)

    return fig


def PlotComparison(y, y_pred):
    wells = [36, 51, 74]
    vmin = min(np.min(y), np.min(y_pred))
    vmax = max(np.max(y), np.max(y_pred))

    fig = make_subplots(rows=1, cols=6, subplot_titles=('Target', 'Prediction', 'Target', 'Prediction', 'Target', 'Prediction'))

    fig.add_trace(go.Heatmap(z=y[wells[0]], zmin=vmin, zmax=vmax, colorscale='Viridis', showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=y_pred[wells[0]], zmin=vmin, zmax=vmax, colorscale='Viridis', showscale=False), row=1, col=2)
    fig.add_trace(go.Heatmap(z=y[wells[1]], zmin=vmin, zmax=vmax, colorscale='Viridis', showscale=False), row=1, col=3)
    fig.add_trace(go.Heatmap(z=y_pred[wells[1]], zmin=vmin, zmax=vmax, colorscale='Viridis', showscale=False), row=1, col=4)
    fig.add_trace(go.Heatmap(z=y[wells[2]], zmin=vmin, zmax=vmax, colorscale='Viridis', showscale=False), row=1, col=5)
    fig.add_trace(go.Heatmap(z=y_pred[wells[2]], zmin=vmin, zmax=vmax, colorscale='Viridis', showscale=True), row=1, col=6)

    fig.update_layout(height=500, width=1000, title_text="Comparison", showlegend=False)
    
    return fig


def PlotR2(y_train, y_pred_train):
    fig, ax = plt.subplots()
    ax.scatter(y_train, y_pred_train, s=0.1)
    plt.plot([0, 1], [0, 1], color='red')
    ax.set_xlabel('Simulated')
    ax.set_ylabel('AI Prediction')
    ax.set_title('R2 Score')
    plt.show()
    

