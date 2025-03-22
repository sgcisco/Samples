from typing import Any
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack

from lambda_core.module_logic.ml.vocabulary import AppVocabulary
from urllib.parse import urlparse
from typing import Dict

from lambda_core.module_logic.ml.clf_dataset import ClassificationDataset
from lambda_core.tools.logger import AppLogger as Logger


def build_app_features(
    df_apps: pd.DataFrame,
    vocabulary: AppVocabulary,
    feature_type: str = "binary",
) -> csr_matrix:
    """Convert pandas application usage into scipy/numpy objects based on the feature type.
    Feature type accepts three values:
    - binary: This returns one-hot encoded vector based on tha application name.
    - total-bytes: This returns a vector based on `totalBytes` per application.
    - server-client-bytes: This returns a vector based on `totalClientBytes` and `totalServerBytes`
      per application.

    Returns
    ------
    Xs
        Sparse feature matrix, with samples as rows and features as columns.
    """
    Xs = _to_scipy_matrix(df_apps.apps, vocabulary, feature_type)

    return Xs


def _to_scipy_matrix(
    series: pd.Series.array, vocabulary: AppVocabulary, feature_type: str
) -> csr_matrix:
    """Converts a Pandas series array where each row is a dict objects into a csr_matrix. The rows
    must include the fields `appName`, `numFlows`, `totalBytes`, totalClientBytes` and
    `totalServerBytes`. The only supported feature type for now is 'binary'.  The resulting
    matrix will contain, for each row, an 1 in the jth column if the corresponding row mentions
    the jth appName in the vocabulary.
    """
    assert feature_type == "binary", "Invalid feature type"

    vector_shape = (1, vocabulary.num_words())

    def _rows_to_vector(rows: pd.Series, feature: str = "numFlows") -> csr_matrix:
        row_matrices = []
        for row in rows:
            if row is None:
                row_matrices.append(csr_matrix(vector_shape, dtype=np.float32))

            csr_j, value_list, out_of_vocab_client, out_of_vocab_server = [], [], [], []

            for r in row:
                if r["appName"] in vocabulary.word_to_index:
                    csr_j.append(vocabulary.word_to_index[r["appName"]])
                    value_list.append(int(r[feature] > 0))
                else:
                    out_of_vocab_client.append(
                        int((r["appName"].startswith("client")) & (r[feature] > 0))
                    )
                    out_of_vocab_server.append(
                        int((r["appName"].startswith("server")) & (r[feature] > 0))
                    )

            csr_j += [
                vocabulary.word_to_index["client-out-of-vocab"],
                vocabulary.word_to_index["server-out-of-vocab"],
            ]
            csr_i = np.zeros(len(csr_j))
            csr_data = value_list + [
                sum(out_of_vocab_client) > 0,
                sum(out_of_vocab_server) > 0,
            ]

            row_matrices.append(
                csr_matrix(
                    (csr_data, (csr_i, csr_j)), shape=vector_shape, dtype=np.bool
                )
            )

        return vstack(row_matrices)

    X = _rows_to_vector(series.array)

    assert X.shape == (len(series), vocabulary.num_words())

    return X


def load_features(
    target_class: str, ref_dataset_version: str, feature_version: str, **kwargs: Any
) -> Dict[str, ClassificationDataset]:
    """The function returns a specific version of the features for the specified reference
    dataset of the target class. The function checks if the desired feature version already
    exists and if yes, it loads them. Otherwise it returns an empty dictionary."""

    # Get Comet experiment
    experiment = kwargs.get("experiment")

    # Find corresponding artifact name for the requested features
    artifact_name = f"features-{target_class}-RefDat-{ref_dataset_version}"

    # Try to load specified version of the artifact. If loading fails, notify the user
    datasets: Dict[str, ClassificationDataset] = {}
    logged_artifact = experiment.get_artifact(  # type: ignore
        artifact_name, version_or_alias=feature_version
    )
    Logger().info("Feature set exists. Loading...")
    for asset in logged_artifact.assets:
        o = urlparse(asset.link, allow_fragments=False)
        key = o.path.lstrip("/")
        bucket = o.netloc
        datasets[asset.logical_path] = ClassificationDataset.load_from_s3(key, bucket)

    return datasets
