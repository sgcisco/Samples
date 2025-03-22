import os
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

import pyspark.sql.functions as F
from pyspark.ml.functions import vector_to_array
from pyspark.ml.stat import Summarizer
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.window import Window
from scipy.sparse import base
from sklearn.cluster import DBSCAN

from clustering.module_logic import distance_matrix, metrics
from clustering.module_logic.cluster_filtering import check_cluster
from clustering.module_logic.constants import (
    _GLOBAL_DISTANCE,
    _PER_ATTRIBUTE_DISTANCE,
    CLUSTER_ASSIGNMENTS_COLUMNS,
    CLUSTER_CHECK_COL,
    DEFAULT_CLUSTERING_PARAMETERS,
    DEFAULT_MIN_SAMPLES,
    OPTIMIZATION_LABEL,
    OPTIMIZATION_METRIC,
    UNASSIGNED_CLUSTER_ID,
)
from clustering.module_logic.distance_matrix import (
    extract_indices_and_weights,
    extract_vectors,
    norm_vectors,
)
from clustering.module_logic.utils import is_optimization_needed, load_clustering_parameters



def _preprocess_snapshot_features(df_features: DataFrame) -> DataFrame:
    """Groups the snapshots by endpoint and collects all relevant feature vectors,
    to the resulting list of feature-vectors.
    """
    assert {"sparseFeatures", "endpoint"}.issubset(df_features.columns)

    df_endpoint_features = (
        df_features.groupBy("endpoint")
        .agg(vector_to_array(Summarizer.sum(F.col("sparseFeatures"))).alias("features"))
        .orderBy(F.col("endpoint"))
    )

    return df_endpoint_features

def _propagate_to_endpoints(
    df_features: DataFrame,
    endpoints: List[str],
    clusters: List[int],
    spark: SparkSession,
) -> DataFrame:
    """Propagates the clusters computed on the features to the endpoints."""
    assert {"endpoint"}.issubset(df_features.columns)

    ref_to_cluster = [
        {"endpoint": ref, "clusterId": int(clu)} for ref, clu in zip(endpoints, clusters)
    ]

    df_ref_cluster = spark.createDataFrame(ref_to_cluster)

    df_endpoints_cluster = df_features.join(df_ref_cluster, on="endpoint", how="left_outer")

    return df_endpoints_cluster


def _get_pd_labels(
    df_snapshot_features_and_labels: DataFrame,
    labels_cols: List[str],
    logger: SparkLogger,
) -> pd.DataFrame:
    """Extracts the labels as a Pandas dataframe."""
    if not df_snapshot_features_and_labels:
        return None

    # Need to associate to each endpoint one label for each dimension.
    # The choice of going for the most recent label, despite arbitrary,
    # is at least deterministic.
    df_labels = (
        df_snapshot_features_and_labels.filter(F.col(OPTIMIZATION_LABEL) != UNKNOWN_LABEL)
        .withColumn(
            "temporalRank",
            F.row_number().over(
                Window.partitionBy("endpoint").orderBy(F.col("snapshotTimestamp").desc())
            ),
        )
        .filter(F.col("temporalRank") == 1)
        .select("endpoint", *labels_cols)
    )

    pd_labels = df_labels.toPandas()

    if pd_labels.empty:
        logger.warn(
            "The given dataframe of labels is empty. No supervised metric will be computed."
        )
        return None

    return pd_labels


def _get_metrics(
    df_endpoint_clusters: DataFrame,
    df_features: DataFrame,
    app_metrics: dict,
    spark: SparkSession,
    logger: SparkLogger,
) -> DataFrame:
    """Computes the metrics and returns them as a Dataframe."""
    pd_endpoints_clusters = df_endpoint_clusters.select("endpoint", "clusterId").toPandas()

    pd_labels = _get_pd_labels(df_features, SUPERVISED_LABELS, logger)
    pd_metrics, pd_per_cluster_metrics = metrics._compute_metrics(
        pd_endpoints_clusters, logger, app_metrics, pd_labels
    )

    df_metrics = spark.createDataFrame(pd_metrics)
    df_per_cluster_metrics = spark.createDataFrame(pd_per_cluster_metrics)

    return df_metrics, df_per_cluster_metrics


def run_optimization(
    df_features: DataFrame,
    map_column_to_weight: dict,
    spark: SparkSession,
    start_ts_ms: int,
    logger: SparkLogger,
    customer_tier: int,
    num_jobs: int = 1,
) -> Tuple[dict, DataFrame]:
    """Wrapper function to update the date of last optimization."""
    logger.info("Running hyper parameter optimization")
    clustering_parameters, df_grid_search_metrics = _hyper_parameter_optimization(
        df_features, map_column_to_weight, spark, logger, customer_tier, num_jobs
    )

    clustering_parameters["optimized_on"] = datetime.utcfromtimestamp(
        start_ts_ms / 1000.0
    ).strftime(DATETIME_FORMAT)

    return clustering_parameters, df_grid_search_metrics


def get_clustering_parameters(
    args,
    start_ts_ms,
    spark,
    df_features,
    map_column_to_weight,
    num_jobs,
    logger,
    customer_tier,
):
    """Retrieve the settings required for running clustering.

    This could either mean loading a current/previously saved setting or running expensive
    hyper parameter optimization.
    """
    df_grid_search_metrics = None
    if is_optimization_needed(start_ts_ms):
        clustering_parameters, df_grid_search_metrics = run_optimization(
            df_features,
            map_column_to_weight,
            spark,
            start_ts_ms,
            logger,
            customer_tier,
            num_jobs,
        )
    else:
        logger.info("Loading cluster parameters from previous run")
        clustering_parameters = load_clustering_parameters(args, start_ts_ms, logger)
        if clustering_parameters is None:
            clustering_parameters, df_grid_search_metrics = run_optimization(
                df_features,
                map_column_to_weight,
                spark,
                start_ts_ms,
                logger,
                customer_tier,
                num_jobs,
            )
        else:
            # Update distance recombination weights
            clustering_parameters["weights"] = map_column_to_weight

    return clustering_parameters, df_grid_search_metrics


def _get_grid_search_param(map_column_to_weight):
    """
    Generate grid search parameters (list) for hyper parameters optimization. DBSCAN parameters
    are in "model_parameters".
    """
    eps_grid_search = [0.001, 0.01] + list(np.arange(0.015, 0.30, 0.05))
    distance_func_grid_search = [
        _GLOBAL_DISTANCE,
        _PER_ATTRIBUTE_DISTANCE,
    ]
    min_samples_grid_search = [DEFAULT_MIN_SAMPLES]

    if "TEST_UUID" in os.environ:
        eps_grid_search = [0.15, 0.01]

    eps_grid_search = [eps for eps in eps_grid_search]

    return [
        {
            "model_parameters": {"eps": eps, "min_samples": min_samples},
            "distance_func": dist_func,
            "weights": map_column_to_weight,
        }
        for eps, min_samples, dist_func in product(
            eps_grid_search, min_samples_grid_search, distance_func_grid_search
        )
    ]


def _hyper_parameter_optimization(
    df_features: DataFrame,
    map_column_to_weight: dict,
    spark: SparkSession,
    logger: SparkLogger,
    customer_tier: int,
    num_jobs: int = 1,
) -> Tuple[dict, DataFrame]:

    # Global reference for normalized vectors
    normed_type_vectors: Dict[str, Tuple[base.spmatrix, np.array, np.array]] = dict()
    # Global reference for distance matrices per distance function and parameters
    raw_distance_matrix: Dict[str, base.spmatrix] = dict()

    best_score = -np.inf

    grid_search = _get_grid_search_param(map_column_to_weight)
    grid_search_metrics: Dict[str, Dict] = {}
    logger.info(f"Running grid search for params {grid_search}")

    start_time = datetime.now()

    df_endpoint_features = _preprocess_snapshot_features(df_features)
    features_vectors, endpoints = extract_vectors(df_endpoint_features, logger)

    feature_df_time = datetime.now()
    feature_time = feature_df_time - start_time
    logger.info(
        f"Features dataframe computed and feature vectors extracted in {feature_time.total_seconds()} "
        f"secs ({feature_time})."
    )

    for clustering_params in grid_search:
        # TODO : Reduce logging once we have a stable version
        if customer_tier != 5 or (
            customer_tier == 5 and clustering_params["distance_func"] == _GLOBAL_DISTANCE
        ):
            df_metrics, _, _, app_metrics = _compute_clusters(
                df_features,
                df_endpoint_features,
                features_vectors,
                endpoints,
                normed_type_vectors,
                raw_distance_matrix,
                clustering_params,
                spark,
                logger,
                customer_tier,
                num_jobs,
            )

            distance_func = clustering_params["distance_func"]
            eps = clustering_params["model_parameters"]["eps"]
            min_samples = clustering_params["model_parameters"]["min_samples"]
            param = f"hpo_{eps}_{min_samples}_{distance_func}"

            for metric_name, metric_value in app_metrics.items():
                tmp_dict = grid_search_metrics.get(metric_name, {})
                tmp_dict[param] = metric_value
                grid_search_metrics[metric_name] = tmp_dict

            pd_metrics = df_metrics.toPandas()

            optimized_metric_value = pd_metrics[pd_metrics.name == OPTIMIZATION_METRIC]["value"]
            logger.info(f"optimized_metric_value {optimized_metric_value} ")

            if optimized_metric_value.empty:
                best_clustering_params = DEFAULT_CLUSTERING_PARAMETERS
                best_clustering_params["weights"] = map_column_to_weight
                break

            if optimized_metric_value.values[0] > best_score:
                best_score = optimized_metric_value.values[0]
                best_clustering_params = clustering_params

    df_pd_grid_search_metrics = pd.DataFrame(grid_search_metrics)
    df_pd_grid_search_metrics["param"] = df_pd_grid_search_metrics.index
    df_pd_grid_search_metrics = df_pd_grid_search_metrics.reset_index(drop=True)
    df_grid_search_metrics = spark.createDataFrame(df_pd_grid_search_metrics)

    return best_clustering_params, df_grid_search_metrics


def _compute_clusters(
    df_features: DataFrame,
    df_endpoint_features: DataFrame,
    features_vectors: list,
    endpoints: list,
    normed_type_vectors: Dict[str, Tuple[base.spmatrix, np.array, np.array]],
    raw_distance_matrix: Dict[str, base.spmatrix],
    clustering_parameters: dict,
    spark: SparkSession,
    logger: SparkLogger,
    customer_tier: int,
    num_jobs: int = 1,
) -> Tuple[DataFrame, DataFrame, DataFrame, dict]:
    """Computes clusters of endpoints based on their features as contained
    in `df_features`, as well as a number of related metrics.

    The main logical steps:
    1. Normalize and deduplicate vectors in `features_vectors` for particular distance function and set of parameters. Cache the results.
       Or extract precomputed values from cache if any is present.
    2. Compute distance matrix for the for particular distance function and set of parameters. Cache the results.
       Or extract precomputed values from cache if any is present and apply parameters.
    3. Run clustering for the set of parameters
    4. Assign cluster labels to endpoints
    5. Compute metrics

    Parameters
    ----------
    df_features
        Dataframe containing the features of the snapshots.
    df_endpoint_features
        Dataframe containing unique endpoints and the associated features over snapshots.
    features_vectors
        Sparse matrix where each row is a feature vector per unique endpoint
    normed_type_vectors
        Global reference for normalized vectors
    raw_distance_matrix
        Global reference for distance matrices per distance function and parameters
    clustering_parameters
        Dictionary with clustering parameters
    spark
        The Spark-Session object.
    logger
        The logger.
    num_jobs
        The number of jobs for DBSCAN.

    Returns
    -------
    Dataframe
        A dataframe with columns `"endpoint"` and `"clusterId"`, mapping
        the endpoints to the ids of their clusters (or UNASSIGNED_CLUSTER_ID,
        when an endpoint is not assigned).
    Dataframe
        A dataframe containing the `name`s and `value`s of the metrics.
    Dataframe
        A dataframe containing the `name`s and `value`s of the per cluster metrics.
    Dictionary
        A dictionary containing all application metrics.
    """

    app_metrics = dict()
    start_time = datetime.now()

    assert hasattr(
        distance_matrix, clustering_parameters["distance_func"]
    ), "Unknown distance function {}".format(clustering_parameters["distance_func"])

    # Get optimized parameters
    matrix_distance_func = getattr(distance_matrix, clustering_parameters["distance_func"])

    logger.info(f"Distance function {clustering_parameters['distance_func']} ")

    start_indices, weights_list = extract_indices_and_weights(
        df_features, clustering_parameters["weights"]
    )

    model_parameters = clustering_parameters["model_parameters"]
    radius = model_parameters["eps"]
    matrix_distance_func_name = clustering_parameters["distance_func"]

    if matrix_distance_func_name not in normed_type_vectors:
        norm_vectors(
            features_vectors, start_indices, matrix_distance_func_name, normed_type_vectors
        )
    features_vectors, features_vectors_inverse_ind, features_vectors_weight = normed_type_vectors[
        matrix_distance_func_name
    ]

    if customer_tier != 5:

        dist_matrix = matrix_distance_func(
            features_vectors,
            raw_distance_matrix,
            radius,
            start_indices,
            weights_list,
            logger,
        )

        distance_matrix_time = datetime.now()
        distance_time = distance_matrix_time - start_time

        logger.info(
            f"Distance matrix generated in {distance_time.total_seconds()} secs ({distance_matrix_time})."
        )

        app_metrics["distMatrixTimeMs"] = int(distance_time.total_seconds() * 1000)

        logger.info(
            "Running clustering algorithm with epsilon={}, min-samples={}".format(
                model_parameters["eps"], model_parameters["min_samples"]
            )
        )
        app_metrics["epsilon"] = model_parameters["eps"]
        app_metrics["minSamples"] = model_parameters["min_samples"]

        model = DBSCAN(**model_parameters, metric="precomputed", n_jobs=num_jobs)
        clusters = model.fit_predict(dist_matrix, sample_weight=features_vectors_weight)
        dist_matrix = None
        clustering_time = datetime.now() - distance_matrix_time
        logger.info(
            f"Clustering proper done in {clustering_time.total_seconds()} secs ({clustering_time})."
        )

        app_metrics["clusteringProperTimeMs"] = int(clustering_time.total_seconds() * 1000)

    elif customer_tier == 5 and matrix_distance_func_name == _GLOBAL_DISTANCE:

        app_metrics["epsilon"] = model_parameters["eps"]
        app_metrics["minSamples"] = model_parameters["min_samples"]

        start_clustering_time = datetime.now()
        # TODO: migrate to HDBSCAN https://hdbscan.readthedocs.io/en/latest/index.html
        model = DBSCAN(**model_parameters, n_jobs=num_jobs)
        clusters = model.fit_predict(features_vectors, sample_weight=features_vectors_weight)
        clustering_time = datetime.now() - start_clustering_time
        logger.info(
            f"Clustering proper done in {clustering_time.total_seconds()} secs ({clustering_time})."
        )

        app_metrics["clusteringProperTimeMs"] = int(clustering_time.total_seconds() * 1000)

    else:
        logger.info(
            f"Clustering for customers with Tier 5 is not applied to functions other then {_GLOBAL_DISTANCE}."
        )

    clusters = clusters[features_vectors_inverse_ind]

    df_endpoint_clusters = _propagate_to_endpoints(df_endpoint_features, endpoints, clusters, spark)
    df_metrics, df_per_cluster_metrics = _get_metrics(
        df_endpoint_clusters, df_features, app_metrics, spark, logger
    )

    return df_metrics, df_endpoint_clusters, df_per_cluster_metrics, app_metrics


def get_endpoint_clusters(
    df_features: DataFrame,
    df_snapshots: DataFrame,
    clustering_parameters: dict,
    spark: SparkSession,
    logger: SparkLogger,
    customer_tier: int,
    num_jobs: int = 1,
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """This is a wrapper over actual clustering function to
    also to assess the quality of clusters.

    Parameters
    ----------
    df_features
        Dataframe containing the features of the snapshots.
    df_snapshots
        Dataframe containing the snapshots.
    clustering_parameters
        Dictionary with clustering parameters
    spark
        The Spark-Session object.
    logger
        The logger.
    num_jobs
        The number of jobs for DBSCAN.

    Returns
    -------
    Dataframe
        A dataframe with columns `"endpoint"` and `"clusterId"`, mapping
        the endpoints to the ids of their clusters (or UNASSIGNED_CLUSTER_ID,
        when an endpoint is not assigned).
    Dataframe
        A dataframe containing distinct rows with `endpoint` and `clusterId` mapping.
    Dataframe
        A dataframe containing the `name`s and `value`s of the metrics.
    Dataframe
        A dataframe containing the `name`s and `value`s of the per cluster metrics.
    """

    # Global reference for normalized vectors
    normed_type_vectors: Dict[str, Tuple[base.spmatrix, np.array, np.array]] = dict()
    # Global reference for distance matrices per distance function and parameters
    raw_distance_matrix: Dict[str, base.spmatrix] = dict()

    logger.info("Best parameters :")
    for param_name, param_value in clustering_parameters.items():
        logger.info("==> {} -- {}".format(param_name, param_value))

    start_time = datetime.now()

    df_endpoint_features = _preprocess_snapshot_features(df_features)
    features_vectors, endpoints = extract_vectors(df_endpoint_features, logger)

    feature_df_time = datetime.now()
    feature_time = feature_df_time - start_time
    logger.info(
        f"Features dataframe computed and feature vectors extracted in {feature_time.total_seconds()} "
        f"secs ({feature_time})."
    )

    start_time = datetime.now()
    (df_metrics, df_endpoint_clusters, df_per_cluster_metrics, app_metrics,) = _compute_clusters(
        df_features,
        df_endpoint_features,
        features_vectors,
        endpoints,
        normed_type_vectors,
        raw_distance_matrix,
        clustering_parameters,
        spark,
        logger,
        customer_tier,
        num_jobs,
    )

    logger.info("Propagating cluster assignments to snapshots...")
    df_clusters = df_features.join(df_endpoint_clusters, how="inner", on="endpoint")

    # Append info regarding quality assesment of the cluster
    df_clusters = check_cluster(df_clusters, df_snapshots)

    df_clusters = df_clusters.select(*CLUSTER_ASSIGNMENTS_COLUMNS, CLUSTER_CHECK_COL)

    # Unassigned devices will have clusterId set to None
    # When generating clusterId uuid, it will be of the form `null-null-5null-nullnull-null`
    # It is made on purpose so that rule_manager can sample negative from it as a "normal" cluster
    # This fake cluster will only be considered as negative sample in rule_manager but NEVER
    # as potential cluster on which a rule need to be extracted
    df_clusters = df_clusters.withColumn(
        "clusterId",
        F.when(F.col("clusterId") == UNASSIGNED_CLUSTER_ID, None).otherwise(F.col("clusterId")),
    )

    total_time = datetime.now() - start_time
    logger.info(f"Total time for clustering: {total_time.total_seconds()} secs ({total_time}).")

    return df_clusters, df_endpoint_clusters, df_metrics, df_per_cluster_metrics
