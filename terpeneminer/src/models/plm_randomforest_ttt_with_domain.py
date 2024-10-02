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
from terpeneminer.src.models.ifaces.domains_sklearn_model import compare_domains_to_known_instances
from terpeneminer.src.models.plm_randomforest import PlmRandomForest
from terpeneminer.src.models.ifaces import BaseConfig
from sklearn.multioutput import MultiOutputClassifier  # type: ignore
from sklearn.preprocessing import MultiLabelBinarizer  # type: ignore
import logging

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


# pylint: disable=R0903, R0901
class PlmRandomForestTTTWithDomains(PlmRandomForest):
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
        with open("data/clustering__domain_dist_based_features.pkl", "rb") as file:
            (
                self.feats_dom_dists,
                self.all_ids_list_dom,
                self.uniid_2_column_ids,
                _,
            ) = pickle.load(file)
        self.allowed_feat_indices: list[int] = None  # type: ignore
        # to experiment with the domain features subset
        # to obtain the subset of domain features, run the following code:
        # python -m src.models.plm_domain_faster.get_domains_feature_importances
        with open("data/domains_subset.pkl", "rb") as file:
            _, self.feat_indices_subset = pickle.load(file)


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

        # comparisons to domains of trn proteins only to avoid leakage
        (
            self.allowed_feat_indices,
            self.dom_features_df,
        ) = compare_domains_to_known_instances(train_df, self, self.feat_indices_subset)
        self.dom_features_df["Emb_dom"] = self.dom_features_df["Emb"]

        features_df = self.features_df_train.merge(
            self.dom_features_df[[self.config.id_col_name, "Emb_dom"]],
            on=self.config.id_col_name,
            how="left",
        )
        missing_dist_feats_bool_idx = features_df["Emb_dom"].isnull()
        features_df.loc[missing_dist_feats_bool_idx, "Emb_dom"] = pd.Series(
            [
                np.zeros(len(self.allowed_feat_indices))
                for _ in range(sum(missing_dist_feats_bool_idx))
            ],
            index=features_df.loc[missing_dist_feats_bool_idx].index,
        )

        features_df["Emb"] = features_df.apply(
            lambda row: np.concatenate((row["Emb"], row["Emb_dom"])), axis=1
        )
        features_df.drop("Emb_dom", axis=1, inplace=True)

        train_data = train_df.merge(features_df, on=self.config.id_col_name)

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
        representations_path_test = f"{self.representations_path_test_base}{getattr(self, 'ttt_rounds')}.h5"
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

        features_df = self.features_df_test.merge(
            self.dom_features_df[[self.config.id_col_name, "Emb_dom"]],
            on=self.config.id_col_name,
            how="left",
        )
        missing_dist_feats_bool_idx = features_df["Emb_dom"].isnull()
        features_df.loc[missing_dist_feats_bool_idx, "Emb_dom"] = pd.Series(
            [
                np.zeros(len(self.allowed_feat_indices))
                for _ in range(sum(missing_dist_feats_bool_idx))
            ],
            index=features_df.loc[missing_dist_feats_bool_idx].index,
        )

        features_df["Emb"] = features_df.apply(
            lambda row: np.concatenate((row["Emb"], row["Emb_dom"])), axis=1
        )
        features_df.drop("Emb_dom", axis=1, inplace=True)

        assert isinstance(val_df, pd.DataFrame), "TTT expects pandas DataFrames"
        val_df = val_df[
            val_df[self.config.target_col_name].map(
                lambda classes: self.config.neg_val not in classes
            )
        ]

        test_df = val_df.merge(
            features_df, on=self.config.id_col_name, copy=False, how="left"
        ).set_index(self.config.id_col_name)
        logger.info(
            "In predict_proba(), features DF shape is: %d x %d",
            *features_df.shape,
        )
        logger.info(
            "In predict_proba(), features dimension is: %d",
            features_df["Emb"].values[0].shape[0],
        )

        average_emb = np.stack(features_df["Emb"].values).mean(axis=0)
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
        return EmbRandomForestTTTConfig
