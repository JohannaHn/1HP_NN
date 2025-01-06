import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import unittest
from unittest.mock import MagicMock, patch
from preprocessing.prepare import prepare_data_and_paths
from data_stuff.utils import SettingsTraining
from main import run
from networks.convLSTM import Seq2Seq 

class TestPrepareDataAndPaths(unittest.TestCase):

    def test_prepare_data_and_paths(self):
        
        # Arrange
        mock_settings = MagicMock()
        mock_settings.dataset_raw = "ep_medium_1000dp_only_vary_dist"
        mock_settings.dataset_prep = "ep_medium_1000dp_only_vary_dist inputs_ks"
        mock_settings.device = "cuda:0"
        mock_settings.epochs = 1
        mock_settings.case = "train"
        mock_settings.model = "default"
        mock_settings.destination = "software_testing"
        mock_settings.inputs = "ks"
        mock_settings.problem = "extend_plumes"
        mock_settings.prev_boxes = 1
        mock_settings.extend = 2
        mock_settings.overfit = 0
        mock_settings.num_layers = 1
        mock_settings.loss = "mse"
        mock_settings.enc_conv_features = [16, 32, 64]
        mock_settings.dec_conv_features = [64, 32, 16, 8]
        mock_settings.enc_kernel_sizes = [5, 5, 5, 5]
        mock_settings.dec_kernel_sizes = [5, 5, 5, 5]
        mock_settings.activation = "relu"
        mock_settings.notes = None
        mock_settings.skip_per_dir = 64

        # Act
        settings = prepare_data_and_paths(mock_settings)

        # Assert
        self.assertIsInstance(settings, SettingsTraining)

class TestRun(unittest.TestCase):

    def test_run(self):
        # Arrange
        mock_settings = MagicMock()
        mock_settings.dataset_raw = "ep_medium_1000dp_only_vary_dist"
        mock_settings.dataset_prep = "ep_medium_1000dp_only_vary_dist inputs_ks"
        mock_settings.device = "cuda:0"
        mock_settings.epochs = 1
        mock_settings.case = "train"
        mock_settings.model = "default"
        mock_settings.destination = "software_testing"
        mock_settings.inputs = "ks"
        mock_settings.problem = "extend_plumes"
        mock_settings.prev_boxes = 1
        mock_settings.extend = 2
        mock_settings.overfit = 0
        mock_settings.num_layers = 1
        mock_settings.loss = "mse"
        mock_settings.enc_conv_features = [16, 32, 64]
        mock_settings.dec_conv_features = [64, 32, 16, 8]
        mock_settings.enc_kernel_sizes = [5, 5, 5, 5]
        mock_settings.dec_kernel_sizes = [5, 5, 5, 5]
        mock_settings.activation = "relu"
        mock_settings.notes = None
        mock_settings.skip_per_dir = 64

        model = run(mock_settings)

        # Assertions
        self.assertIsInstance(model, Seq2Seq)

if __name__ == "__main__":
    unittest.main()