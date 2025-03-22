from itertools import tee
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pyspark.sql.functions as F
from clustering.module_logic.constants import _GLOBAL_DISTANCE, _PER_ATTRIBUTE_DISTANCE
from kairosflow.spark_pipeline import SparkLogger
from kronos.utils.attribute_utils import get_attribute_vocabulary
from pyspark.sql import DataFrame
from pyspark.sql.types import IntegerType
from scipy.sparse import base, coo_matrix, csr_matrix, hstack, vstack
from scipy.sparse.linalg import norm
from sklearn.metrics.pairwise import linear_kernel


def get_udf_vector_size():
    """
    Returns size of the vector.
    """

    def _get_vector_size(v):
        return v.size

    return F.udf(_get_vector_size, IntegerType())


def pairwise(iterable: Iterable[Any]) -> Iterable[Any]:
    """pairwise('ABCDEFG') --> AB BC CD DE EF FG
    In our case it is the slice on the columns
    [(0, 129), (129, 131), (131, 140), ... )]
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def extract_indices_and_weights(
    df_features: DataFrame, weights: Dict[str, float]
) -> Tuple[List[int], List[float]]:
    """Extract start_indices and weights for each attribute"""
    start_indices = []
    cur_index = 0
    weights_list = []

    for col_name, vocabulary in get_attribute_vocabulary(df_features):
        cur_index += len(vocabulary)
        start_indices.append(cur_index)
        weights_list.append(weights[col_name])

    return start_indices, weights_list


def extract_vectors(
    df_endpoint_features: DataFrame, logger: SparkLogger
) -> Tuple[base.spmatrix, List[str]]:
    """
    Extract all Spark Sparse vectors and endpoints. Creates the Scipy sparse matrix from vectors
    and returns the number of snapshots that a feature has been present for a specific endpoint
    """
    rows, endpoints = zip(
        *[
            (r[0], r[1])
            for r in df_endpoint_features.select("features", "endpoint").toLocalIterator()
        ]
    )
    features_vectors = vstack([csr_matrix(row) for row in rows])
    return features_vectors, endpoints


def norm_vectors(
    features_vectors: base.spmatrix,
    start_indices: List[int],
    matrix_distance_func_name: str,
    normed_type_vectors: Dict[str, Tuple[base.spmatrix, np.array, np.array]],
) -> None:
    """Compute norm values for the feature matrix on the row/ unique endpoint basis and cache the results.
    In case of `_compute_per_attribute_distance_matrix`:
        1. Derive slices representing range of columns per each feature.
        2. For each particular slice:
            1. Slice features_vectors to select columns representing a particular feature
            2. Compute norms per each row/attribute
            3. Normalize each row/attribute
        3. Horizontally stack all normalized sliced matrices and deduplicate vectors and keep weight indices and inverse indices for reconstruction
        4. Cache results for further use
    In case of `_compute_global_distance_matrix`:
        1. Compute norms per each row/attribute
        2. Normalize each row/attribute
        3. Deduplicate vectors and keep weight indices and inverse indices for reconstruction
        4. Cache results for further use
    """

    def _calculate_norm(mtrx):
        return norm(mtrx, axis=1)

    if matrix_distance_func_name == _PER_ATTRIBUTE_DISTANCE:

        slices = list(pairwise([0] + start_indices))

        normed_sub_vectors = []
        for s in slices:
            # Cosine similarity on the sparse vectors
            sub_vectors = features_vectors[:, s[0] : s[1]]
            sub_vectors_ecd = csr_matrix(norm(sub_vectors, axis=1)).T
            sub_vectors_ecd.data = 1 / sub_vectors_ecd.data
            sub_vectors_norm = sub_vectors.multiply(sub_vectors_ecd)
            normed_sub_vectors.append(sub_vectors_norm)

        normed_sub_vectors = hstack(normed_sub_vectors, format="csr")
        normed_sub_vectors_lil = normed_sub_vectors.tolil()
        (
            _,
            features_vectors_ind,
            features_vectors_inverse_ind,
            features_vectors_weight,
        ) = np.unique(
            normed_sub_vectors_lil.data + normed_sub_vectors_lil.rows,
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )

        normed_sub_vectors = normed_sub_vectors[features_vectors_ind]

        normed_type_vectors[matrix_distance_func_name] = (
            normed_sub_vectors,
            features_vectors_inverse_ind,
            features_vectors_weight,
        )

    elif matrix_distance_func_name == _GLOBAL_DISTANCE:
        features_vectors_ecd = csr_matrix(_calculate_norm(features_vectors)).T
        features_vectors_ecd.data = 1 / features_vectors_ecd.data
        global_features_vectors_norm = features_vectors.multiply(features_vectors_ecd)

        global_features_vectors_norm_lil = global_features_vectors_norm.tolil()
        (
            _,
            features_vectors_ind,
            features_vectors_inverse_ind,
            features_vectors_weight,
        ) = np.unique(
            global_features_vectors_norm_lil.data + global_features_vectors_norm_lil.rows,
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )

        global_features_vectors_norm = global_features_vectors_norm[features_vectors_ind]
        normed_type_vectors[matrix_distance_func_name] = (
            global_features_vectors_norm,
            features_vectors_inverse_ind,
            features_vectors_weight,
        )

    else:
        raise ValueError("Matrix distance function is not set")


def _compute_per_attribute_distance_matrix(
    feature_vectors: base.spmatrix,
    raw_distance_matrix: Dict[str, base.spmatrix],
    radius: float,
    start_indices: List[int],
    weights: List[float],
    logger: SparkLogger,
) -> base.spmatrix:
    """Compute distance matrix on the attribute basis.
    1. Compute distance matrix per each attribute by selecting the relevant subset of columns
    2. Multiply attribute distance matrix on the weight of the relevant attribute
    3. Sum all attribute matrices and normalize it over sum of weights
    4. Filters all distance below the radius
    """

    func_param_reference = f"{_PER_ATTRIBUTE_DISTANCE}_{'_'.join(map(str, weights))}"

    if func_param_reference not in raw_distance_matrix:

        slices = list(pairwise([0] + start_indices))
        dist_matrix = csr_matrix(
            (feature_vectors.shape[0], feature_vectors.shape[0]), dtype=np.float32
        )

        for s, w in zip(slices, weights):
            # Cosine similarity on the sparse vectors
            sub_vectors = feature_vectors[:, s[0] : s[1]]
            # linear_kernel is used as vectors are normalized at this point
            sub_mtrx_dist = linear_kernel(sub_vectors, dense_output=False)
            # Find all zero vectors
            mask = np.nonzero(np.isclose(sub_vectors.sum(axis=1), 0))
            # Zero - zero vectors interactions require distances to be set to 0 across all submatrices
            # Thus similarities have to be set to 1 explicitly
            # 1. Find all zero vectors
            rows_to_cols = mask[0]
            # 2. Get all rows via `np.repeat` and cols `np.tile` references on zero to zero interactions
            # if rows_to_cols = [1 2 3]
            # then np.repeat(rows_to_cols, rows_to_cols.size) = [1 1 1 2 2 2 3 3 3]
            # then np.tile(rows_to_cols, rows_to_cols.size) = [1 2 3 1 2 3 1 2 3]
            # np.column_stack(...).astype("int").T returns
            # array([[1, 1, 1, 2, 2, 2, 3, 3, 3],
            #        [1, 2, 3, 1, 2, 3, 1, 2, 3]])
            index_list_transpose = (
                np.column_stack(
                    (
                        np.repeat(rows_to_cols, rows_to_cols.size),
                        np.tile(rows_to_cols, rows_to_cols.size),
                    )
                )
                .astype("int")
                .T
            )
            # 3. Set 1 at the inds of interest by ...
            _zz_row = index_list_transpose[0]
            _zz_col = index_list_transpose[1]
            _zz_data = np.ones(len(index_list_transpose[0]))
            # 4. ... constructing a coo matrix
            zz_sub_mtrx_dist = coo_matrix(
                (_zz_data, (_zz_row, _zz_col)),
                shape=(sub_vectors.shape[0], sub_vectors.shape[0]),
                dtype=np.int32,
            )
            # 4. Convert this matrix to CSR and sum with current distance matrix
            # sum of two csr matrices is much faster then to set values directly for the CSR
            sub_mtrx_dist += zz_sub_mtrx_dist.tocsr()
            sub_mtrx_dist.data = sub_mtrx_dist.data * w
            # Total sum
            dist_matrix += sub_mtrx_dist

        # Normalizing by all weights and calculate distance
        dist_matrix.data = 1 - dist_matrix.data / sum(weights)
        # Check and nullify all small values below zero due to arithmetic operations
        assert np.amin(dist_matrix.data) > -1e-05
        dist_matrix.data = np.maximum(dist_matrix.data, 0)
        # Removing all noise and very small values to avoid impact of float arithmetics
        dist_matrix.data = np.around(dist_matrix.data, decimals=4)
        raw_distance_matrix[func_param_reference] = dist_matrix.astype(np.float32, copy=False)

    # Nulling all values below certain distance
    dist_matrix = raw_distance_matrix[func_param_reference].copy()
    dist_matrix.data[dist_matrix.data >= radius] = 0
    dist_matrix.eliminate_zeros()
    return dist_matrix


def _compute_global_distance_matrix(
    feature_vectors: base.spmatrix,
    raw_distance_matrix: Dict[str, base.spmatrix],
    radius: float,
    *args: Any,
) -> base.spmatrix:
    """Compute distance matrix on the global basis and removes all distances below radius value"""
    if _GLOBAL_DISTANCE not in raw_distance_matrix:
        # Global similarity
        dist_matrix = linear_kernel(feature_vectors, dense_output=False)
        dist_matrix.data = 1 - dist_matrix.data
        # Check and nullify all small values below zero due to arithmetic operations
        assert np.amin(dist_matrix.data) > -1e-05
        dist_matrix.data = np.maximum(dist_matrix.data, 0)
        # Removing all noise and very small values to avoid impact of float arithmetics
        dist_matrix.data = np.around(dist_matrix.data, decimals=4)
        # Global distance
        raw_distance_matrix[_GLOBAL_DISTANCE] = dist_matrix.astype(np.float32, copy=False)

    dist_matrix = raw_distance_matrix[_GLOBAL_DISTANCE].copy()
    # Nulling all values below certain distance
    dist_matrix.data[dist_matrix.data >= radius] = 0
    dist_matrix.eliminate_zeros()
    return dist_matrix
