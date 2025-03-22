from typing import List, Dict, Any
import numpy as np
import lightgbm as lgbm
from sklearn.metrics import f1_score
from lambda_core.module_logic.ml.helper import StratifiedGroupKFold
import pytorch_lightning as pl
from sklearn.base import BaseEstimator
import torch.nn as nn


class FeatureLearner:
    """
    This class handles the training of deep learning model to learn features of legit and attack
    set
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def fit(
        self,
        dataloaders: Dict[str, Any],
        epochs: int = 30,
        callbacks: List = [],
        num_gpus: int = 1,
    ) -> None:
        """
        The function trains feature learner model using PyTorch Lightning

        Parameters
        ----------
        - datasets: Dictionary containing torch dataset loader per split.
        epochs: Number of epochs to train.
        - callbacks: Contains config for training parameters like EarlyStopping.
        - num_gpus: Number of GPUs to be used for training.
        """
        trainer = pl.Trainer(max_epochs=epochs, callbacks=callbacks, gpus=num_gpus)
        trainer.fit(
            model=self.model,
            train_dataloaders=dataloaders["train"],
            val_dataloaders=dataloaders["test"],
        )


class GradientBoosting:
    """
    This class handles training of Gradient Boosting model
    """

    def __init__(self, model: BaseEstimator):
        self.clf = model

    def fit(
        self,
        X: np.array,
        y: np.array,
        groups: np.array,
        sample_weight: np.array,
        params: Dict[str, Any],
    ) -> None:

        """
        The function trains gradient boosting model using lightGBM.

        Parameters
        ----------
        - X: Input vector containing features generated from trained Feature Learner Model.
        - y: Labels for samples.
        - groups: Contains list of mac addresses used for StratifiedGroupKFold split.
        - sample_weight: Weights of the samples.
        - params: Contains two parameters `model_params` and `optuna_params`. If `model_params` is
        set, the parameters are used for training skips hyper parameter optimization. If not set,
        the model undergoes hyperparameter optimization process using `optuna_params`.
        """
        if params["model_params"] is None:
            print("Running hyper parameter optimization")
            params["model_params"] = self.get_best_params(
                X, y, groups, sample_weight, params["optuna_params"]
            )
        self.clf = self.clf.set_params(**params["model_params"])
        self.clf.fit(X, y, sample_weight)

    def get_best_params(
        self,
        X: np.array,
        y: np.array,
        groups: np.array,
        sample_weight: np.array,
        optuna_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """

        Runs hyperparameter optimization using Optuna and returns best parameters.

        Parameters
        ----------
        - X: Input vector containing features generated from trained Feature Learner Model.
        - y: Labels for samples.
        - groups: Contains list of mac addresses used for StratifiedGroupKFold split.
        - sample_weight: Weights of the samples.
        - optuna_params: Model parameter grid to explore.
        """
        import optuna
        from optuna.integration import LightGBMPruningCallback

        def objective(trial, X, y, groups, sample_weight, optuna_params):

            param_grid = {
                "n_estimators": trial.suggest_categorical(
                    "n_estimators", optuna_params["n_estimators"]
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    optuna_params["learning_rate"][0],
                    optuna_params["learning_rate"][1],
                    step=optuna_params["learning_rate"][2],
                ),
                "num_leaves": trial.suggest_int(
                    "num_leaves",
                    optuna_params["num_leaves"][0],
                    optuna_params["num_leaves"][1],
                    step=optuna_params["num_leaves"][2],
                ),
                "max_depth": trial.suggest_categorical(
                    "max_depth", optuna_params["max_depth"]
                ),
                "min_data_in_leaf": trial.suggest_int(
                    "min_data_in_leaf",
                    optuna_params["min_data_in_leaf"][0],
                    optuna_params["min_data_in_leaf"][1],
                    step=optuna_params["min_data_in_leaf"][2],
                ),
                "lambda_l1": trial.suggest_float(
                    "lambda_l1",
                    optuna_params["lambda_l1"][0],
                    optuna_params["lambda_l1"][1],
                    step=optuna_params["lambda_l1"][2],
                ),
                "lambda_l2": trial.suggest_float(
                    "lambda_l2",
                    optuna_params["lambda_l2"][0],
                    optuna_params["lambda_l2"][1],
                    step=optuna_params["lambda_l2"][2],
                ),
                "min_gain_to_split": trial.suggest_float(
                    "min_gain_to_split",
                    optuna_params["min_gain_to_split"][0],
                    optuna_params["min_gain_to_split"][1],
                    step=optuna_params["min_gain_to_split"][2],
                ),
            }

            cv = StratifiedGroupKFold(n_splits=optuna_params["n_splits"], shuffle=False)
            cv_scores = np.empty(optuna_params["n_splits"])
            for idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
                X_train_, X_test_ = X[train_idx], X[test_idx]
                y_train_, y_test_ = y[train_idx], y[test_idx]

                model = lgbm.LGBMClassifier(**param_grid)
                model.fit(
                    X_train_,
                    y_train_,
                    eval_set=[(X_test_, y_test_)],
                    eval_metric="auc",
                    early_stopping_rounds=optuna_params["early_stopping_rounds"],
                    callbacks=[LightGBMPruningCallback(trial, "auc")],
                    verbose=-1,
                    sample_weight=sample_weight[train_idx],
                )
                preds = model.predict(X_test_)
                cv_scores[idx] = f1_score(
                    y_test_, preds, sample_weight=sample_weight[test_idx]
                )

            return np.mean(cv_scores)

        study = optuna.create_study(direction="maximize", study_name="LGBM Classifier")
        func = lambda trial: objective(  # noqa
            trial,
            X=X,
            y=y,
            groups=groups,
            sample_weight=sample_weight,
            optuna_params=optuna_params,
        )
        study.optimize(func, n_trials=optuna_params["n_trials"])

        return study.best_params
