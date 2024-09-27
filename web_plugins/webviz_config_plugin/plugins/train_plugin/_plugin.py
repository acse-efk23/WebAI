from dash.development.base_component import Component

from webviz_config import WebvizPluginABC

from ._callbacks import plugin_callbacks
from ._layout import main_layout

from deepdown import *
import os


class Train(WebvizPluginABC):

    def __init__(self):

        super().__init__(stretch=True)

        print(f"Current working directory of Train is: {os.getcwd()}")

        self.train_dataset = Hdf5Dataset()

        self.validation_dataset = Hdf5Dataset()

        self.ml_operator = MLOperator()

        self.set_callbacks()

    @property
    def layout(self):
        return main_layout(get_uuid=self.uuid)

    def set_callbacks(self):
        plugin_callbacks(self.uuid, self.train_dataset, self.validation_dataset, self.ml_operator)
