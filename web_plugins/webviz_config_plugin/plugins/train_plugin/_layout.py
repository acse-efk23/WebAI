from typing import Callable, List

import webviz_core_components as wcc

from dash import html
from dash import dcc

import plotly.graph_objs as go

######################################################################
#
# Collection of Dash layout and ID-ownership
#
# Defines the Dash layout, and builds the HTML-element structure
# for the plugin.
#
# Ownership of the layout element ID's, which is provided to the
# various callback Inputs, States and Outputs.
#
######################################################################


class LayoutElements:

    BATCH_SIZE = "batch-size"                           # dcc.Input()
    EPOCHS = "epochs"                                   # dcc.Input()
    LOAD_BUTTON = "load-dataset-button"                 # html.Button()
    TRAIN_BUTTON = "train-button"                       # html.Button()

    PRINT_DATASET = "print-dataset"                     # html.Div()
    PRINT_TRAIN = "print-train"                         # html.Div()
    INTERVAL = "interval"                               # dcc.Interval()
    LOSS_GRAPH = "loss-graph"                           # dcc.Graph()
    PREDICTION_GRAPH = "prediction-graph"               # dcc.Graph()



def main_layout(get_uuid: Callable):
    
    return wcc.FlexBox(
        children=[
            wcc.FlexColumn(
                children=wcc.Frame(
                    style={"height": "90vh"},
                    children=[

                        wcc.Selectors(
                            label="Advanced AI Settings",
                            children=[

                                html.Div([

                                    ####################################################

                                    # Select batch size
                                    html.Div([
                                        html.Label('Enter Batch Size:', style={'font-weight': 'bold'}),
                                        dcc.Input(
                                            id=get_uuid(LayoutElements.BATCH_SIZE),
                                            value=1
                                        )
                                    ]),

                                    # Select epochs
                                    html.Div([
                                        html.Label('Enter Epochs:', style={'font-weight': 'bold'}),
                                        dcc.Input(
                                            id=get_uuid(LayoutElements.EPOCHS),
                                            value=50
                                        )
                                    ]),

                                    ####################################################

                                    # Interval
                                    dcc.Interval(id=get_uuid(LayoutElements.INTERVAL), interval=100, n_intervals=0, disabled=True),

                                ], style={'padding': '5px'}),

                            ]
                        ),

                        # Button hdf5 dataset
                        html.Div([
                            html.Button(children=['Load Training Data'], id=get_uuid(LayoutElements.LOAD_BUTTON), n_clicks=0,
                                        style={'margin-top': '20px'})
                        ]),

                        # Button train
                        html.Div([
                            html.Button(children=['Train'], id=get_uuid(LayoutElements.TRAIN_BUTTON), n_clicks=0,
                                        style={'margin-top': '20px'}),
                        ]),

                    ],
                )
            ),

            wcc.FlexColumn(
                flex=4,
                children=[
                    
                    # Print dataset result
                    html.Div([
                        html.Div(children=[html.Pre("Please click the button to load hdf5 file.", 
                                                    id=get_uuid(LayoutElements.PRINT_DATASET))],
                                #  style={'display': 'none'}
                                 ),
                    ]),
                    
                    # Print training result
                    html.Div([
                        html.Div(children=[html.Pre("Please click the button to train the model.", 
                                                    id=get_uuid(LayoutElements.PRINT_TRAIN))],
                                 style={'display': 'block'}
                                ),
                    ]),

                    # Graph to plot live training loss
                    dcc.Graph(id=get_uuid(LayoutElements.LOSS_GRAPH), 
                                figure=go.Figure(
                                    data=[go.Scatter(x=[], y=[])],
                                    layout=go.Layout(
                                        title="Training Loss Over Epochs",
                                        xaxis=dict(title="Epoch"),
                                        yaxis=dict(title="Loss"),
                                        template="plotly_white"
                                    )
                                ),
                                style={'height': '300px', 'width': '100%'}
                    ),

                    # Graph to plot prediction
                    dcc.Graph(id=get_uuid(LayoutElements.PREDICTION_GRAPH), 
                                figure=go.Figure(
                                    data=[go.Scatter(x=[], y=[])],
                                    layout=go.Layout(
                                        title="Predicted Pressure Field",
                                        xaxis=dict(title="Epoch"),
                                        yaxis=dict(title="Loss"),
                                        template="plotly_white"
                                    )
                                ),
                                style={'height': '400px', 'width': '100%'}
                    ),
    

                ],
            ),
        ],
    )
