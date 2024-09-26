import numpy as np
import torch

from deepdown import *


class EvaluatorClass:
    """
    EvaluatorClass is designed to evaluate a machine learning model's performance on training and validation datasets.
    It computes various error metrics such as R-squared (R2) and Relative Absolute Error (RAE) for both datasets.
    """

    def __init__(self, model, train_loader, validation_loader):

        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self.target_train = None
        self.prediction_train = None
        self.target_validation = None
        self.prediction_validation = None

        self.r2_list_train = []
        self.r2_avg_train = 0
        self.r2_std_train = 0
        self.r2_list_validation = []
        self.r2_avg_validation = 0
        self.r2_std_validation = 0

        self.rae_train = 0
        self.rae_validation = 0

    def error_metrics(self):
        """
        Calculate and store error metrics for training and validation datasets.
        This method computes the R-squared (R2) and Relative Absolute Error (RAE)
        metrics for both the training and validation datasets. The predictions are
        made using the `pressure_field` method, and the metrics are calculated
        using the `r2_score_numpy` and `rae_score_numpy` functions.
        The following attributes are updated:
        - r2_list_train: List of R2 scores for the training dataset.
        - rae_train: Average RAE score for the training dataset.
        - r2_avg_train: Average R2 score for the training dataset.
        - r2_std_train: Standard deviation of R2 scores for the training dataset.
        - r2_list_validation: List of R2 scores for the validation dataset.
        - rae_validation: Average RAE score for the validation dataset.
        - r2_avg_validation: Average R2 score for the validation dataset.
        - r2_std_validation: Standard deviation of R2 scores for the validation dataset.
        """

        self.pressure_field()

        for pred, targ in zip(self.prediction_train, self.target_train):
            self.r2_list_train.append(r2_score_numpy(pred, targ))
            self.rae_train += rae_score_numpy(pred, targ)
        self.rae_train /= len(self.target_train)
        self.r2_avg_train = sum(self.r2_list_train) / len(self.r2_list_train)
        self.r2_std_train = np.std(self.r2_list_train)

        for pred, targ in zip(
            self.prediction_validation, self.target_validation
        ):
            self.r2_list_validation.append(r2_score_numpy(pred, targ))
            self.rae_validation += rae_score_numpy(pred, targ)
        self.rae_validation /= len(self.target_validation)
        self.r2_avg_validation = sum(self.r2_list_validation) / len(
            self.r2_list_validation
        )
        self.r2_std_validation = np.std(self.r2_list_validation)

    def pressure_field(self):
        """
        Computes and stores the pressure field predictions for both the training and validation datasets.
        This method sets the model to evaluation mode and makes predictions on the training and validation
        datasets using the provided data loaders. The predictions and targets are then scaled and stored
        as attributes of the class instance.
        """
        self.model.eval()

        target_list = []
        prediction_list = []
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader):
                input, target = batch
                prediction = self.model(input)
                target_list.append(target)
                prediction_list.append(prediction)
        target_train = torch.cat(target_list, dim=0)
        prediction_train = torch.cat(prediction_list, dim=0)
        self.target_train = target_train.numpy() * (950 - 195) + 195
        self.prediction_train = prediction_train.numpy() * (950 - 195) + 195

        target_list = []
        prediction_list = []
        with torch.no_grad():
            for i, batch in enumerate(self.validation_loader):
                input, target = batch
                prediction = self.model(input)
                target_list.append(target)
                prediction_list.append(prediction)
        target_validation = torch.cat(target_list, dim=0)
        prediction_validation = torch.cat(prediction_list, dim=0)
        self.target_validation = target_validation.numpy() * (950 - 195) + 195
        self.prediction_validation = (
            prediction_validation.numpy() * (950 - 195) + 195
        )
