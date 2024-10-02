"""A class for Random Forest predictive models on top of protein language model (PLM) embeddings"""
import pickle
from typing import Type, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import os

from terpeneminer.src.models.config_classes import (
    EmbRandomForestConfig,
    EmbMLPConfig,
    EmbLogisticRegressionConfig,
EmbRandomForestTTTConfig
)
from terpeneminer.src.models.plm_randomforest import PlmRandomForest
from terpeneminer.src.models.ifaces import BaseConfig
from sklearn.multioutput import MultiOutputClassifier  # type: ignore
from sklearn.preprocessing import MultiLabelBinarizer  # type: ignore
import logging

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


# pylint: disable=R0903, R0901
class PlmRandomForestTPSClassification(PlmRandomForest):
    """
    Random Forest on top of protein language model (PLM) embeddings with TTT
    """

    def __init__(
        self,
        config: EmbRandomForestTTTConfig | EmbRandomForestConfig | EmbMLPConfig | EmbLogisticRegressionConfig,
    ):
        super().__init__(
            config=config,
        )


    def fit_core(self, train_df: pd.DataFrame, class_name: str = None):
        train_df = train_df[
            train_df[self.config.target_col_name].map(
                lambda classes: self.config.neg_val not in classes
            )
        ]

        assert (
            self.features_df is not None
        ), "self.features_df has not been initialized!"
        logger.info("In fit(), features DF shape is: %d x %d", *self.features_df.shape)
        logger.info(
            "In fit(), features dimension is: %d",
            self.features_df["Emb"].values[0].shape[0],
        )
        train_data = train_df.merge(self.features_df, on=self.config.id_col_name)

        if not self.per_class_optimization:
            try:
                requires_multioutputwrapper_for_multilabel = (
                    self.config.requires_multioutputwrapper_for_multilabel  # type: ignore
                )
            except AttributeError:
                requires_multioutputwrapper_for_multilabel = False
            if requires_multioutputwrapper_for_multilabel:
                self.classifier = MultiOutputClassifier(
                    self.classifier_class(**self.get_model_specific_params())
                )
            else:
                self.classifier = self.classifier_class(
                    **self.get_model_specific_params()
                )
            label_binarizer = MultiLabelBinarizer(classes=self.config.class_names)
            logger.info("Preparing multi-labels...")
            target = label_binarizer.fit_transform(
                train_data[self.config.target_col_name].values
            )
            logger.info("Fitting the model...")

            self.classifier.fit(
                np.stack(train_data["Emb"].values),
                target,
            )
            self.config.class_names = label_binarizer.classes_
        else:
            raise ValueError(
                    f"No per-class optimization for TTT."
                )

    def predict_proba(
            self,
            val_df: pd.DataFrame | np.ndarray,
            selected_class_name: Optional[str] = None,
    ) -> np.ndarray:

        assert isinstance(val_df, pd.DataFrame), "TTT expects pandas DataFrames"
        val_df = val_df[
            val_df[self.config.target_col_name].map(
                lambda classes: self.config.neg_val not in classes
            )
        ]

        test_df = val_df.merge(
            self.features_df, on=self.config.id_col_name, copy=False, how="left"
        ).set_index(self.config.id_col_name)
        logger.info(
            "In predict_proba(), features DF shape is: %d x %d",
            *self.features_df.shape,
        )
        logger.info(
            "In predict_proba(), features dimension is: %d",
            self.features_df["Emb"].values[0].shape[0],
        )

        average_emb = np.stack(self.features_df["Emb"].values).mean(axis=0)
        test_df["Emb"] = test_df["Emb"].map(
            lambda x: x if isinstance(x, np.ndarray) else average_emb
        )
        test_df = test_df.loc[val_df[self.config.id_col_name]]
        test_embs_np = np.stack(test_df["Emb"].values)
        try:
            per_class_optimization = self.config.per_class_optimization
        except AttributeError:
            per_class_optimization = False
        val_proba_np = np.zeros((len(val_df), len(self.config.class_names)))
        if not per_class_optimization:
            logger.info(
                "Global model %s predicts proba for %d samples (%d features each)",
                str(self.classifier),
                *test_embs_np.shape,
            )
            y_pred_proba = self.classifier.predict_proba(test_embs_np)
            for class_i in range(len(self.config.class_names)):
                val_proba_np[:, class_i] = (
                    y_pred_proba[class_i][:, -1]
                    if isinstance(y_pred_proba, list)
                    else y_pred_proba[:, class_i]
                )
        else:
            raise ValueError("TTT does not support per-class optimization")

        return val_proba_np

    @classmethod
    def config_class(cls) -> Type[BaseConfig]:
        """
        A getter of the model-specific config class
        :return:  A dataclass for config storage
        """
        return EmbRandomForestConfig
