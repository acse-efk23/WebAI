from dash.development.base_component import Component

from webviz_config import WebvizPluginABC

from ._callbacks import plugin_callbacks
from ._layout import main_layout

from deepdown import SimDataset
import os


class LoadData(WebvizPluginABC):

    def __init__(self):

        super().__init__(stretch=True)

        print(f"Current working directory of LoadData is: {os.getcwd()}")

        self.sim_dataset = SimDataset()

        self.set_callbacks()

    @property
    def layout(self):
        return main_layout(get_uuid=self.uuid)

    def set_callbacks(self):
        plugin_callbacks(self.uuid, self.sim_dataset)
