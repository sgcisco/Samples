from datetime import timedelta
from typing import List, Tuple
from logging import Logger
import pandas as pd

from lambda_core.definitions.iana.iana import iana_ports

from lambda_core.definitions.constants import COLUMNS_AGG_MAX, COLUMNS_QUASI_DUPLICATE
from lambda_core.definitions.network_reference.network_reference import prepare_network_reference_taxonomy_apps
from lambda_core.definitions.constants import (
    UNKNOWN_APP_NAME,
    UNKNOWN_APP_NAMES,
    LOGICAL_CLASS_KEY,
)


def deduplicate_flows(nf_pd: pd.DataFrame) -> pd.DataFrame:
    # Drop duplicate and quasi-duplicate flows
    nf_pd.drop_duplicates(subset=COLUMNS_QUASI_DUPLICATE, inplace=True)

    nf_pd = (
        nf_pd.groupby(COLUMNS_AGG_MAX, dropna=False, sort=False)
        .agg(
            clientBytesSent=("clientBytesSent", "max"),
            serverBytesSent=("serverBytesSent", "max"),
            clientPacketsSent=("clientPacketsSent", "max"),
            serverPacketsSent=("serverPacketsSent", "max"),
            timestamp=("timestamp", "max"),
            clientMacDCS=("clientMacDCS", "first"),
            serverMacDCS=("serverMacDCS", "first"),
        )
        .reset_index()
    )

    return nf_pd


def replace_custom_network_reference_appnames(nf_pd: pd.DataFrame) -> pd.DataFrame:
    df_network_reference_protos_pd = prepare_network_reference_taxonomy_apps()

    nf_pd = nf_pd.merge(
        df_network_reference_protos_pd,
        on=["appName"],
        how="left",
        copy=False,
    ).fillna({"isInNETWORK_REFERENCETaxonomy": False})
    # can we use them to better specify these matchings?
    auto_application_condition = nf_pd["appName"].str.contains(
        r"^.*:\d+(?:_(?:UDP|TCP))?$", regex=True
    )

    over_http_condition = nf_pd["appName"].str.contains("over-http")
    custom_application_condition = nf_pd["appName"].str.contains(".", regex=False)

    tcp_proto_condition = nf_pd["ipProto"] == 6
    http_port_condition = nf_pd["serverPort"] == 80
    https_port_condition = nf_pd["serverPort"] == 443

    nf_pd.loc[
        tcp_proto_condition & (nf_pd["appName"].str.startswith("http-")),
        ["appName", "isInNETWORK_REFERENCETaxonomy"],
    ] = ("http", True)
    nf_pd.loc[
        tcp_proto_condition & (nf_pd["appName"].str.startswith("ssl-")),
        ["appName", "isInNETWORK_REFERENCETaxonomy"],
    ] = ("ssl", True)
    # For <>-over-http traffic:
    nf_pd.loc[
        over_http_condition & tcp_proto_condition, ["appName", "isInNETWORK_REFERENCETaxonomy"]
    ] = (
        nf_pd.loc[over_http_condition & tcp_proto_condition]["appName"]
        .str.split("over-http")
        .str.get(0)
        .apply(lambda x: f"{x}over-http"),
        True,
    )
    # Should never happen but it needs to be marked as unknown
    nf_pd.loc[
        over_http_condition & ~tcp_proto_condition, ["appName", "isInNETWORK_REFERENCETaxonomy"]
    ] = (UNKNOWN_APP_NAME, True)
    # Or some another protocol we don't recognize:
    nf_pd.loc[
        auto_application_condition & tcp_proto_condition & https_port_condition,
        ["appName", "isInNETWORK_REFERENCETaxonomy"],
    ] = ("https", True)
    nf_pd.loc[
        auto_application_condition & tcp_proto_condition & http_port_condition,
        ["appName", "isInNETWORK_REFERENCETaxonomy"],
    ] = ("http", True)
    nf_pd.loc[
        ~nf_pd["isInNETWORK_REFERENCETaxonomy"] & auto_application_condition,
        ["appName", "isInNETWORK_REFERENCETaxonomy"],
    ] = (UNKNOWN_APP_NAME, True)
    # The remainder are either protocols defined by the customer
    # or SNI/DN found by network_reference in the SSL traffic
    nf_pd.loc[
        custom_application_condition & tcp_proto_condition & https_port_condition,
        ["appName", "isInNETWORK_REFERENCETaxonomy"],
    ] = ("https", True)
    nf_pd.loc[
        custom_application_condition & tcp_proto_condition & http_port_condition,
        ["appName", "isInNETWORK_REFERENCETaxonomy"],
    ] = ("http", True)
    nf_pd.loc[
        ~nf_pd["isInNETWORK_REFERENCETaxonomy"] & tcp_proto_condition & custom_application_condition,
        ["appName", "isInNETWORK_REFERENCETaxonomy"],
    ] = ("ssl", True)
    nf_pd.loc[
        ~nf_pd["isInNETWORK_REFERENCETaxonomy"] & auto_application_condition,
        ["appName", "isInNETWORK_REFERENCETaxonomy"],
    ] = (UNKNOWN_APP_NAME, True)
    nf_pd.loc[~nf_pd["isInNETWORK_REFERENCETaxonomy"], "appName"] = UNKNOWN_APP_NAME

    nf_pd.drop(columns=["isInNETWORK_REFERENCETaxonomy"], inplace=True)

    return nf_pd


def with_port_based_apps(nf_pd: pd.DataFrame) -> pd.DataFrame:
    """Augment flow dataframe by reclassifying unknown applications based on IANA ports."""
    iana_ports_pd = iana_ports()[["ipProto", "port", "appName"]].rename(
        columns={"appName": "appNameIANA", "port": "serverPort"}
    )
    iana_ports_pd = iana_ports_pd[~iana_ports_pd["serverPort"].str.contains("-")]
    iana_ports_pd["serverPort"] = iana_ports_pd["serverPort"].astype("int64")
    target_cols = nf_pd.columns
    unkhown_apps_ind = nf_pd["appName"].isin(UNKNOWN_APP_NAMES)
    nf_pd = nf_pd.merge(
        iana_ports_pd, on=["ipProto", "serverPort"], how="left", copy=False
    )
    nf_pd.loc[unkhown_apps_ind, "appName"] = nf_pd.loc[unkhown_apps_ind, "appNameIANA"]
    nf_pd.dropna(subset=["appName"], inplace=True)
    assert "unknown" not in nf_pd["appName"].values
    assert "Unknown" not in nf_pd["appName"].values

    return nf_pd[target_cols]


def flows_to_netflow(
    nf_pd: pd.DataFrame,
    endpoint_pd: pd.DataFrame,
    active_classes: List[str],
    logger: Logger,
) -> pd.DataFrame:
    """Keeps only known traffic for the known endpoint points and adds the information from `endpoint_pd`.
    Column `logicalClass` is added to `nf_pd` to indicate whether the logical class of the
    client/server endpoint is known and to keep traffic for the active classes only.
    """
    # 1. Consolidate ``clientMac``, ``serverMac`` based on DCS MAC address.
    nf_pd["clientMac"] = nf_pd["clientMac"].combine_first(nf_pd["clientMacDCS"])
    nf_pd["serverMac"] = nf_pd["serverMac"].combine_first(nf_pd["serverMacDCS"])
    nf_pd.drop(columns=["clientMacDCS", "serverMacDCS"], inplace=True)

    # 2. Resolve netflows with endpoint
    common_cols = [
        "timestamp",
        "ipProto",
        "serverPort",
        "appName",
        "clientBytesSent",
        "serverBytesSent",
    ]
    server_nf_pd = nf_pd[common_cols + ["serverMac"]].rename(
        columns={"serverMac": "mac"}
    )
    server_nf_pd["side"] = "server"
    client_nf_pd = nf_pd[common_cols + ["clientMac"]].rename(
        columns={"clientMac": "mac"}
    )
    client_nf_pd["side"] = "client"
    nf_pd = pd.concat(
        [server_nf_pd, client_nf_pd], copy=False, ignore_index=True, sort=False
    )
    logger.info(f"Input raw NF has {len(nf_pd)} rows")

    nf_pd = nf_pd[nf_pd["mac"].notnull()]
    logger.info(f"NF after null MAC removal has {len(nf_pd)} rows")

    endpoint_pd = endpoint_pd.reset_index()[["macAddress", LOGICAL_CLASS_KEY, "uuid"]].rename(
        columns={"macAddress": "mac"}
    )

    nf_pd.set_index("mac", inplace=True)
    endpoint_pd.set_index("mac", inplace=True)

    nf_with_logical_class_pd = nf_pd.merge(
        endpoint_pd, on="mac", how="inner", copy=False
    ).reset_index()
    logger.info(
        f"NF after removing rows with no logical class has"
        f" {len(nf_with_logical_class_pd)} rows"
    )

    nf_with_logical_class_pd = nf_with_logical_class_pd[
        nf_with_logical_class_pd[LOGICAL_CLASS_KEY].isin(active_classes)
    ]

    return nf_with_logical_class_pd


def flows_agg_inference_interval(
    nf_with_logical_class_pd: pd.DataFrame,
    inference_interval: timedelta,
    logger: Logger,
) -> pd.DataFrame:

    nf_with_logical_class_pd["timeBin"] = nf_with_logical_class_pd[
        "timestamp"
    ].dt.floor("{}min".format(inference_interval.total_seconds() / 60))
    nf_app_flow_count_pd = (
        nf_with_logical_class_pd.groupby(
            ["timeBin", "mac", "appName", "side", "uuid", LOGICAL_CLASS_KEY],
            dropna=False,
            sort=False,
        )
        .agg(
            clientBytesInterval=("clientBytesSent", "sum"),
            serverBytesInterval=("serverBytesSent", "sum"),
            numFlowsInterval=("appName", "size"),
        )
        .reset_index()
        .rename(columns={LOGICAL_CLASS_KEY: "label"})
    )

    nf_app_flow_count_pd["totalBytesInterval"] = nf_app_flow_count_pd[
        ["clientBytesInterval", "serverBytesInterval"]
    ].sum(axis=1)
    nf_app_flow_count_pd["appName"] = (
        nf_app_flow_count_pd["side"] + "-" + nf_app_flow_count_pd["appName"]
    )

    return nf_app_flow_count_pd


def flows_agg_window(
    completed_bins_pd: pd.DataFrame,
    window_size: timedelta,
    logger: Logger,
) -> pd.DataFrame:

    nf_app_agg_window_interval_pd = (
        completed_bins_pd.groupby(
            ["mac", "appName", "side", "label", "windowStart", "windowEnd", "uuid"],
            dropna=False,
            sort=False,
        )
        .agg(
            clientBytesWindow=("clientBytesInterval", "sum"),
            serverBytesWindow=("serverBytesInterval", "sum"),
            totalBytesWindow=("totalBytesInterval", "sum"),
            numFlowsWindow=("numFlowsInterval", "sum"),
        )
        .reset_index()
    )

    return nf_app_agg_window_interval_pd


def agg_flows_to_list(nf_app_agg_window_interval_pd: pd.DataFrame) -> pd.DataFrame:

    nf_app_agg_window_interval_pd.rename(
        columns={
            "numFlowsWindow": "numFlows",
            "totalBytesWindow": "totalBytes",
            "clientBytesWindow": "totalClientBytes",
            "serverBytesWindow": "totalServerBytes",
        },
        inplace=True,
    )

    nf_app_agg_window_interval_pd["apps"] = nf_app_agg_window_interval_pd[
        ["appName", "totalClientBytes", "totalServerBytes", "totalBytes", "numFlows"]
    ].apply(dict, axis=1)

    nf_agg_flows_to_list = (
        nf_app_agg_window_interval_pd.groupby(
            ["mac", "label", "windowStart", "windowEnd", "uuid"],
            dropna=False,
            sort=False,
        )["apps"]
        .agg(list)
        .reset_index()
    )

    return nf_agg_flows_to_list


def build_apps_usage(
    nf_with_logical_class_pd: pd.DataFrame,
    etl_pd: pd.DataFrame,
    inference_interval: timedelta,
    window_size: timedelta,
    logger: Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    last_record_timestamp = nf_with_logical_class_pd["timestamp"].max()
    # 1. Calculate aggregation values over the current inference interval
    nf_app_agg_inference_interval_pd = flows_agg_inference_interval(
        nf_with_logical_class_pd, inference_interval, logger
    )
    # We can compute features only for the time bins for which all of the traffic
    # has been observed. Therefore we only keep the records belonging to time bins
    # whose end timestamp is prior than the latest record timestamp.
    ret_etl_pd = pd.concat(
        [nf_app_agg_inference_interval_pd, etl_pd],
        copy=False,
        ignore_index=True,
        sort=False,
    )

    ret_etl_pd["timeBinEnd"] = ret_etl_pd["timeBin"] + pd.to_timedelta(
        inference_interval.total_seconds(), unit="s"
    )

    ret_etl_pd["isCompleteBin"] = ret_etl_pd["timeBinEnd"] <= last_record_timestamp
    #     ret_etl_pd["isToUpdate"] = ret_etl_pd["timeBin"] >= first_record_timestamp
    if not ret_etl_pd[ret_etl_pd["isCompleteBin"]].empty:
        # End of the latest complete bin
        window_end = ret_etl_pd[ret_etl_pd["isCompleteBin"]]["timeBinEnd"].max()
        logger.info(f"Latest completed bin {window_end}")

        window_start = max(
            window_end - window_size,
            ret_etl_pd[ret_etl_pd["isCompleteBin"]]["timeBin"].min(),
        )

        logger.info(f"Window processing between {window_start} {window_end}")
        ret_etl_pd["windowStart"] = window_start
        ret_etl_pd["windowEnd"] = window_end
        ret_etl_pd = ret_etl_pd[ret_etl_pd["timeBin"] >= window_start]
        # 2. Calculate aggregation values over the current sliding window
        # Make sure we return only the bins falling into the sliding window
        completed_bins_pd = ret_etl_pd[ret_etl_pd["isCompleteBin"]]
        nf_app_agg_window_interval_pd = flows_agg_window(
            completed_bins_pd, window_size, logger
        )

        nf_agg_flows_to_list = agg_flows_to_list(nf_app_agg_window_interval_pd)
    else:
        logger.info("No NF aggregated completed bins")
        nf_agg_flows_to_list = pd.DataFrame()
    return (
        nf_agg_flows_to_list,
        nf_app_agg_inference_interval_pd,
    )


def create_dataset(
    nf_pd: pd.DataFrame,
    endpoint_pd: pd.DataFrame,
    etl_pd: pd.DataFrame,
    active_classes: List[str],
    inference_interval: timedelta,
    window_size: timedelta,
    logger: Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    last_record_timestamp = nf_pd["timestamp"].max()
    first_record_timestamp = nf_pd["timestamp"].min()

    logger.info(
        f"Dataset creation: first netflow record timestamp {first_record_timestamp}, last record "
        f"timestamp {last_record_timestamp}"
    )
    logger.info(f"Input raw NF: {len(nf_pd)} rows")
    # Feature construction - Flows deduplication
    logger.info("Deduplicate flows")
    nf_pd = deduplicate_flows(nf_pd)
    logger.info(f"Input NF size after flow deduplication: {len(nf_pd)} rows")
    # Resolve netflows with endpoint
    nf_with_logical_class_pd = flows_to_netflow(nf_pd, endpoint_pd, active_classes, logger)
    logger.info(
        f"Input NF size after class assignment: {len(nf_with_logical_class_pd)} rows"
    )
    # Feature construction - NETWORK_REFERENCE application resolutions
    nf_with_logical_class_pd = replace_custom_network_reference_appnames(nf_with_logical_class_pd)
    logger.info(
        f"Input NF size after NETWORK_REFERENCE resolution: {len(nf_with_logical_class_pd)} rows"
    )
    # Feature construction - IANA application resolutions
    nf_with_logical_class_pd = with_port_based_apps(nf_with_logical_class_pd)
    logger.info(
        f"Input NF size after IANA resolution: {len(nf_with_logical_class_pd)} rows"
    )

    # At this point, the dataframe could be empty because of port scan, no
    # logical class or no MAC address. If this is the case, stop processing
    if nf_with_logical_class_pd.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Run df_apps construction
    df_apps, ret_etl_pd = build_apps_usage(
        nf_with_logical_class_pd,
        etl_pd,
        inference_interval,
        window_size,
        logger,
    )

    return df_apps, ret_etl_pd
