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
class PlmRandomForestTTT(PlmRandomForest):
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
        self.features_df_train = self.features_df.copy()
        self.representations_path_test_base: str = config.representations_path_test
        self.classifier = None


    def fit_core(self, train_df: pd.DataFrame, class_name: str = None):
        train_df = train_df[
            train_df[self.config.target_col_name].map(
                lambda classes: self.config.neg_val not in classes
            )
        ]

        assert (
            self.features_df_train is not None
        ), "self.features_df_train has not been initialized!"
        logger.info("In fit(), features DF shape is: %d x %d", *self.features_df_train.shape)
        logger.info(
            "In fit(), features dimension is: %d",
            self.features_df_train["Emb"].values[0].shape[0],
        )
        train_data = train_df.merge(self.features_df_train, on=self.config.id_col_name)
        logger.info(
            "train_data size is: %d",
            len(train_data),
        )

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
            print('target: ', target)
            logger.info("Fitting the model...")

            self.classifier.fit(
                np.stack(train_data["Emb"].values),
                target,
            )
            self.config.class_names = label_binarizer.classes_
            for class_i, class_name in enumerate(self.config.class_names):
                print("In train: ", class_name, ": ", target[:, class_i].mean())
        else:
            raise ValueError(
                    f"No per-class optimization for TTT."
                )

    def predict_proba(
            self,
            val_df: pd.DataFrame | np.ndarray,
            selected_class_name: Optional[str] = None,
    ) -> np.ndarray:
        representations_path_test = f"{self.representations_path_test_base}{getattr(self, 'ttt_rounds')}.h5"
        print('representation path: ', f"{self.representations_path_test_base}{getattr(self, 'ttt_rounds')}.h5")
        if not os.path.exists(representations_path_test):
            raise FileNotFoundError(
                f"File with pre-computed embeddings {representations_path_test} does not exist"
            )
        self.features_df_test: pd.DataFrame = pd.read_hdf(representations_path_test)
        assert isinstance(self.features_df, pd.DataFrame), (
            "Expected embeddings to be stored as pandas Dataframe"
            f"But {type(self.features_df_test)} found at {representations_path_test}"
        )
        self.features_df_test.drop_duplicates(subset=[self.config.id_col_name], inplace=True)
        self.features_df_test.columns = [self.config.id_col_name, "Emb"]

        assert isinstance(val_df, pd.DataFrame), "TTT expects pandas DataFrames"
        val_df = val_df[
            val_df[self.config.target_col_name].map(
                lambda classes: self.config.neg_val not in classes
            )
        ]

        test_df = val_df.merge(
            self.features_df_test, on=self.config.id_col_name, copy=False, how="left"
        ).set_index(self.config.id_col_name)
        logger.info(
            "In predict_proba(), features DF shape is: %d x %d",
            *self.features_df_test.shape,
        )
        logger.info(
            "In predict_proba(), features dimension is: %d",
            self.features_df_test["Emb"].values[0].shape[0],
        )

        average_emb = np.stack(self.features_df_test["Emb"].values).mean(axis=0)
        logger.info("Size of test: %d", len(test_df))
        logger.info("number of present test embeddings: %d", test_df["Emb"].map(lambda x: isinstance(x, np.ndarray)).sum())
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
            # for class_i in range(len(self.config.class_names)):
            for class_i, class_name in enumerate(self.config.class_names):
                val_proba_np[:, class_i] = (
                    y_pred_proba[class_i][:, -1]
                    if isinstance(y_pred_proba, list)
                    else y_pred_proba[:, class_i]
                )
                print("In predict: ", class_name, ": ", val_proba_np[:, class_i].mean())
        else:
            raise ValueError("TTT does not support per-class optimization")

        return val_proba_np

    @classmethod
    def config_class(cls) -> Type[BaseConfig]:
        """
        A getter of the model-specific config class
        :return:  A dataclass for config storage
        """
        return EmbRandomForestTTTConfig
