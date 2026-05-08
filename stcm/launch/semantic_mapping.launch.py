import os
from pathlib import Path

import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.logging import get_logger
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


DEFAULT_TEXT_PROMPT = "table . door . chair ."
DEFAULT_GRAPH_PATH = "stcm.json"
DEFAULT_USE_SIM_TIME = False
DEFAULT_RUN_UPDATER = False
DEFAULT_RESET_TF_ON_TIME_JUMP = True

try:
    DEFAULT_CONFIG_PATH = os.path.join(
        get_package_share_directory("stcm"),
        "config",
        "semantic_mapping_params.yaml",
    )
except PackageNotFoundError:
    DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "config" / "semantic_mapping_params.yaml")


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "on")


def _load_config(config_path):
    resolved_path = os.path.expanduser(config_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Config file '{resolved_path}' does not exist.")
    with open(resolved_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_str(context, arg_name, fallback):
    override = LaunchConfiguration(arg_name).perform(context)
    return override if override else fallback


def _resolve_bool(context, arg_name, fallback):
    override = LaunchConfiguration(arg_name).perform(context)
    if override:
        return _parse_bool(override)
    return _parse_bool(fallback)


def _resolve_float(context, arg_name, fallback):
    override = LaunchConfiguration(arg_name).perform(context)
    if override:
        return float(override)
    return float(fallback)


def _resolve_int(context, arg_name, fallback):
    override = LaunchConfiguration(arg_name).perform(context)
    if override:
        return int(override)
    return int(fallback)


def launch_setup(context, *args, **kwargs):
    config_path = LaunchConfiguration("config_file").perform(context)
    config = _load_config(config_path)

    def _node_params(drop_keys=None):
        params = dict(config)
        params.pop("run_updater", None)
        if drop_keys:
            for key in drop_keys:
                params.pop(key, None)
        return params

    text_prompt = _resolve_str(context, "text_prompt", config.get("text_prompt", DEFAULT_TEXT_PROMPT))
    graph_path = _resolve_str(context, "graph_output_path", config.get("graph_output_path", DEFAULT_GRAPH_PATH))
    use_sim_time = _resolve_bool(context, "use_sim_time", config.get("use_sim_time", DEFAULT_USE_SIM_TIME))
    run_updater = _resolve_bool(context, "run_updater", config.get("run_updater", DEFAULT_RUN_UPDATER))
    offline_sequential = _resolve_bool(
        context,
        "offline_sequential",
        config.get("offline_sequential", False),
    )
    offline_frame_stride = _resolve_int(
        context,
        "offline_frame_stride",
        config.get("offline_frame_stride", 1),
    )
    reset_tf_on_time_jump = _resolve_bool(
        context,
        "reset_tf_on_time_jump",
        config.get("reset_tf_on_time_jump", DEFAULT_RESET_TF_ON_TIME_JUMP),
    )
    edge_distance_threshold = _resolve_float(
        context,
        "edge_distance_threshold",
        config.get("edge_distance_threshold", 3.0),
    )
    gng_enabled = _resolve_bool(context, "gng_enabled", config.get("gng_enabled", False))
    gng_per_label = _resolve_bool(context, "gng_per_label", config.get("gng_per_label", True))
    gng_max_nodes = _resolve_int(context, "gng_max_nodes", config.get("gng_max_nodes", 1000))
    gng_lambda = _resolve_int(context, "gng_lambda", config.get("gng_lambda", 200))
    gng_max_age = _resolve_int(context, "gng_max_age", config.get("gng_max_age", 200))
    gng_eps_w = _resolve_float(context, "gng_eps_w", config.get("gng_eps_w", 0.05))
    gng_eps_n = _resolve_float(context, "gng_eps_n", config.get("gng_eps_n", 0.0006))
    gng_alpha = _resolve_float(context, "gng_alpha", config.get("gng_alpha", 0.95))
    gng_beta = _resolve_float(context, "gng_beta", config.get("gng_beta", 0.9995))
    gng_min_obs = _resolve_int(
        context,
        "gng_min_observations_to_commit",
        config.get("gng_min_observations_to_commit", 3),
    )
    gng_cluster_merge_distance = _resolve_float(
        context,
        "gng_cluster_merge_distance",
        config.get("gng_cluster_merge_distance", 0.5),
    )
    gng_outlier_gate_meters = _resolve_float(
        context,
        "gng_outlier_gate_meters",
        config.get("gng_outlier_gate_meters", 0.0),
    )
    max_observation_range_m = _resolve_float(
        context,
        "max_observation_range_m",
        config.get("max_observation_range_m", 0.0),
    )
    place_gng_enabled = _resolve_bool(
        context,
        "place_gng_enabled",
        config.get("place_gng_enabled", False),
    )
    place_gng_distance_threshold = _resolve_float(
        context,
        "place_gng_distance_threshold",
        config.get("place_gng_distance_threshold", 1.5),
    )
    place_gng_eps_w = _resolve_float(
        context,
        "place_gng_eps_w",
        config.get("place_gng_eps_w", 0.1),
    )
    place_gng_eps_n = _resolve_float(
        context,
        "place_gng_eps_n",
        config.get("place_gng_eps_n", 0.01),
    )
    place_gng_max_edge_age = _resolve_int(
        context,
        "place_gng_max_edge_age",
        config.get("place_gng_max_edge_age", 50),
    )
    place_gng_max_nodes = _resolve_int(
        context,
        "place_gng_max_nodes",
        config.get("place_gng_max_nodes", 0),
    )
    place_gng_lambda = _resolve_int(
        context,
        "place_gng_lambda",
        config.get("place_gng_lambda", 100),
    )
    place_gng_alpha = _resolve_float(
        context,
        "place_gng_alpha",
        config.get("place_gng_alpha", 0.95),
    )
    place_gng_beta = _resolve_float(
        context,
        "place_gng_beta",
        config.get("place_gng_beta", 0.9995),
    )
    place_gng_semantic_alpha = _resolve_float(
        context,
        "place_gng_semantic_alpha",
        config.get("place_gng_semantic_alpha", 0.1),
    )
    place_gng_semantic_aggregation = _resolve_str(
        context,
        "place_gng_semantic_aggregation",
        config.get("place_gng_semantic_aggregation", "max"),
    )
    place_gng_use_second_best_edge = _resolve_bool(
        context,
        "place_gng_use_second_best_edge",
        config.get("place_gng_use_second_best_edge", True),
    )
    place_gng_use_transition_edges = _resolve_bool(
        context,
        "place_gng_use_transition_edges",
        config.get("place_gng_use_transition_edges", True),
    )
    place_gng_update_when_empty = _resolve_bool(
        context,
        "place_gng_update_when_empty",
        config.get("place_gng_update_when_empty", False),
    )
    place_gng_input_path = _resolve_str(
        context,
        "place_gng_input_path",
        config.get("place_gng_input_path", graph_path),
    )
    place_gng_output_path = _resolve_str(
        context,
        "place_gng_output_path",
        config.get("place_gng_output_path", graph_path),
    )
    grounding_ckpt = _resolve_str(context, "groundingdino_checkpoint", config.get("groundingdino_checkpoint", ""))
    mobilesam_ckpt = _resolve_str(context, "mobilesam_checkpoint", config.get("mobilesam_checkpoint", ""))
    depth_ckpt = _resolve_str(context, "depth_anything_checkpoint", config.get("depth_anything_checkpoint", ""))
    use_depth_anything_fallback = _resolve_bool(
        context,
        "use_depth_anything_fallback",
        config.get("use_depth_anything_fallback", True),
    )
    rosbag_path = _resolve_str(context, "rosbag_path", config.get("rosbag_path", ""))
    rosbag_storage_id = _resolve_str(
        context,
        "rosbag_storage_id",
        config.get("rosbag_storage_id", "sqlite3"),
    )

    if offline_sequential and run_updater:
        logger = get_logger("semantic_mapping.launch")
        logger.warning("offline_sequential is enabled; disabling run_updater for this launch.")
        run_updater = False

    builder_params = _node_params(drop_keys=["place_gng_input_path"])
    builder_params.update(
        {
            "text_prompt": text_prompt,
            "graph_output_path": graph_path,
            "use_sim_time": use_sim_time,
            "edge_distance_threshold": edge_distance_threshold,
            "gng_enabled": gng_enabled,
            "gng_per_label": gng_per_label,
            "gng_max_nodes": gng_max_nodes,
            "gng_lambda": gng_lambda,
            "gng_max_age": gng_max_age,
            "gng_eps_w": gng_eps_w,
            "gng_eps_n": gng_eps_n,
            "gng_alpha": gng_alpha,
            "gng_beta": gng_beta,
            "gng_min_observations_to_commit": gng_min_obs,
            "gng_cluster_merge_distance": gng_cluster_merge_distance,
            "gng_outlier_gate_meters": gng_outlier_gate_meters,
            "max_observation_range_m": max_observation_range_m,
            "place_gng_enabled": place_gng_enabled,
            "place_gng_distance_threshold": place_gng_distance_threshold,
            "place_gng_eps_w": place_gng_eps_w,
            "place_gng_eps_n": place_gng_eps_n,
            "place_gng_max_edge_age": place_gng_max_edge_age,
            "place_gng_max_nodes": place_gng_max_nodes,
            "place_gng_lambda": place_gng_lambda,
            "place_gng_alpha": place_gng_alpha,
            "place_gng_beta": place_gng_beta,
            "place_gng_semantic_alpha": place_gng_semantic_alpha,
            "place_gng_semantic_aggregation": place_gng_semantic_aggregation,
            "place_gng_use_second_best_edge": place_gng_use_second_best_edge,
            "place_gng_use_transition_edges": place_gng_use_transition_edges,
            "place_gng_update_when_empty": place_gng_update_when_empty,
            "place_gng_output_path": place_gng_output_path,
            "reset_tf_on_time_jump": reset_tf_on_time_jump,
            "groundingdino_checkpoint": grounding_ckpt,
            "mobilesam_checkpoint": mobilesam_ckpt,
            "depth_anything_checkpoint": depth_ckpt,
            "use_depth_anything_fallback": use_depth_anything_fallback,
            "offline_sequential": offline_sequential,
            "offline_frame_stride": offline_frame_stride,
            "rosbag_path": rosbag_path,
            "rosbag_storage_id": rosbag_storage_id,
        }
    )

    builder_node = Node(
        package="stcm",
        executable="semantic_map_builder",
        name="semantic_map_builder",
        output="screen",
        parameters=[builder_params],
    )

    if run_updater:
        updater_params = _node_params(
            drop_keys=[
                "offline_sequential",
                "offline_frame_stride",
                "rosbag_path",
                "rosbag_storage_id",
            ]
        )
        updater_params.update(
            {
                "text_prompt": text_prompt,
                "graph_input_path": config.get("graph_input_path", graph_path),
                "graph_output_path": graph_path,
                "use_sim_time": use_sim_time,
            "edge_distance_threshold": edge_distance_threshold,
            "gng_enabled": gng_enabled,
            "gng_per_label": gng_per_label,
            "gng_max_nodes": gng_max_nodes,
            "gng_lambda": gng_lambda,
            "gng_max_age": gng_max_age,
            "gng_eps_w": gng_eps_w,
            "gng_eps_n": gng_eps_n,
            "gng_alpha": gng_alpha,
                "gng_beta": gng_beta,
                "gng_min_observations_to_commit": gng_min_obs,
                "gng_cluster_merge_distance": gng_cluster_merge_distance,
                "gng_outlier_gate_meters": gng_outlier_gate_meters,
            "max_observation_range_m": max_observation_range_m,
                "place_gng_enabled": place_gng_enabled,
                "place_gng_distance_threshold": place_gng_distance_threshold,
                "place_gng_eps_w": place_gng_eps_w,
                "place_gng_eps_n": place_gng_eps_n,
                "place_gng_max_edge_age": place_gng_max_edge_age,
                "place_gng_max_nodes": place_gng_max_nodes,
                "place_gng_lambda": place_gng_lambda,
                "place_gng_alpha": place_gng_alpha,
                "place_gng_beta": place_gng_beta,
                "place_gng_semantic_alpha": place_gng_semantic_alpha,
                "place_gng_semantic_aggregation": place_gng_semantic_aggregation,
                "place_gng_use_second_best_edge": place_gng_use_second_best_edge,
                "place_gng_use_transition_edges": place_gng_use_transition_edges,
                "place_gng_update_when_empty": place_gng_update_when_empty,
                "place_gng_input_path": place_gng_input_path,
                "place_gng_output_path": place_gng_output_path,
                "reset_tf_on_time_jump": reset_tf_on_time_jump,
            "groundingdino_checkpoint": grounding_ckpt,
            "mobilesam_checkpoint": mobilesam_ckpt,
                "depth_anything_checkpoint": depth_ckpt,
                "use_depth_anything_fallback": use_depth_anything_fallback,
            }
        )

        updater_node = Node(
            package="stcm",
            executable="semantic_map_updater",
            name="semantic_map_updater",
            output="screen",
            parameters=[updater_params],
        )
        return [builder_node, updater_node]

    return [builder_node]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=DEFAULT_CONFIG_PATH,
                description="Path to a YAML file with semantic mapping parameters.",
            ),
            DeclareLaunchArgument(
                "text_prompt",
                default_value="",
                description="Override the prompt from the config file.",
            ),
            DeclareLaunchArgument(
                "graph_output_path",
                default_value="",
                description="Override the STCM graph path from the config file.",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="",
                description="Override the use_sim_time flag from the config file.",
            ),
            DeclareLaunchArgument(
                "run_updater",
                default_value="",
                description="Override the run_updater flag from the config file.",
            ),
            DeclareLaunchArgument(
                "offline_sequential",
                default_value="",
                description="Override the offline_sequential flag from the config file.",
            ),
            DeclareLaunchArgument(
                "offline_frame_stride",
                default_value="",
                description="Override the offline_frame_stride value from the config file.",
            ),
            DeclareLaunchArgument(
                "reset_tf_on_time_jump",
                default_value="",
                description="Override the TF buffer reset-on-time-jump flag from the config file.",
            ),
            DeclareLaunchArgument(
                "edge_distance_threshold",
                default_value="",
                description="Override the edge distance threshold from the config file.",
            ),
            DeclareLaunchArgument(
                "gng_enabled",
                default_value="",
                description="Override the gng_enabled flag from the config file.",
            ),
            DeclareLaunchArgument(
                "gng_per_label",
                default_value="",
                description="Override the gng_per_label flag from the config file.",
            ),
            DeclareLaunchArgument(
                "gng_max_nodes",
                default_value="",
                description="Override the gng_max_nodes value from the config file.",
            ),
            DeclareLaunchArgument(
                "gng_lambda",
                default_value="",
                description="Override the gng_lambda value from the config file.",
            ),
            DeclareLaunchArgument(
                "gng_max_age",
                default_value="",
                description="Override the gng_max_age value from the config file.",
            ),
            DeclareLaunchArgument(
                "gng_eps_w",
                default_value="",
                description="Override the gng_eps_w value from the config file.",
            ),
            DeclareLaunchArgument(
                "gng_eps_n",
                default_value="",
                description="Override the gng_eps_n value from the config file.",
            ),
            DeclareLaunchArgument(
                "gng_alpha",
                default_value="",
                description="Override the gng_alpha value from the config file.",
            ),
            DeclareLaunchArgument(
                "gng_beta",
                default_value="",
                description="Override the gng_beta value from the config file.",
            ),
            DeclareLaunchArgument(
                "gng_min_observations_to_commit",
                default_value="",
                description="Override the gng_min_observations_to_commit value from the config file.",
            ),
            DeclareLaunchArgument(
                "gng_cluster_merge_distance",
                default_value="",
                description="Override the gng_cluster_merge_distance value from the config file.",
            ),
            DeclareLaunchArgument(
                "gng_outlier_gate_meters",
                default_value="",
                description="Override the gng_outlier_gate_meters value from the config file.",
            ),
            DeclareLaunchArgument(
                "max_observation_range_m",
                default_value="",
                description="Reject detections whose 3D pose is farther than this (meters) from the robot base xy. 0 disables.",
            ),
            DeclareLaunchArgument(
                "place_gng_enabled",
                default_value="",
                description="Override the place_gng_enabled flag from the config file.",
            ),
            DeclareLaunchArgument(
                "place_gng_distance_threshold",
                default_value="",
                description="Override the place_gng_distance_threshold value from the config file.",
            ),
            DeclareLaunchArgument(
                "place_gng_eps_w",
                default_value="",
                description="Override the place_gng_eps_w value from the config file.",
            ),
            DeclareLaunchArgument(
                "place_gng_eps_n",
                default_value="",
                description="Override the place_gng_eps_n value from the config file.",
            ),
            DeclareLaunchArgument(
                "place_gng_max_edge_age",
                default_value="",
                description="Override the place_gng_max_edge_age value from the config file.",
            ),
            DeclareLaunchArgument(
                "place_gng_max_nodes",
                default_value="",
                description="Override the place_gng_max_nodes value from the config file.",
            ),
            DeclareLaunchArgument(
                "place_gng_lambda",
                default_value="",
                description="Override the place_gng_lambda value from the config file.",
            ),
            DeclareLaunchArgument(
                "place_gng_alpha",
                default_value="",
                description="Override the place_gng_alpha value from the config file.",
            ),
            DeclareLaunchArgument(
                "place_gng_beta",
                default_value="",
                description="Override the place_gng_beta value from the config file.",
            ),
            DeclareLaunchArgument(
                "place_gng_semantic_alpha",
                default_value="",
                description="Override the place_gng_semantic_alpha value from the config file.",
            ),
            DeclareLaunchArgument(
                "place_gng_semantic_aggregation",
                default_value="",
                description="Override the place_gng_semantic_aggregation value from the config file.",
            ),
            DeclareLaunchArgument(
                "place_gng_use_second_best_edge",
                default_value="",
                description="Override the place_gng_use_second_best_edge value from the config file.",
            ),
            DeclareLaunchArgument(
                "place_gng_use_transition_edges",
                default_value="",
                description="Override the place_gng_use_transition_edges value from the config file.",
            ),
            DeclareLaunchArgument(
                "place_gng_update_when_empty",
                default_value="",
                description="Override the place_gng_update_when_empty value from the config file.",
            ),
            DeclareLaunchArgument(
                "place_gng_input_path",
                default_value="",
                description="Override the place_gng_input_path value from the config file.",
            ),
            DeclareLaunchArgument(
                "place_gng_output_path",
                default_value="",
                description="Override the place_gng_output_path value from the config file (set to the STCM path to embed).",
            ),
            DeclareLaunchArgument(
                "groundingdino_checkpoint",
                default_value="",
                description="Override the GroundingDINO checkpoint path from the config file.",
            ),
            DeclareLaunchArgument(
                "mobilesam_checkpoint",
                default_value="",
                description="Override the MobileSAM checkpoint path from the config file.",
            ),
            DeclareLaunchArgument(
                "depth_anything_checkpoint",
                default_value="",
                description="Override the Depth-Anything model path from the config file.",
            ),
            DeclareLaunchArgument(
                "use_depth_anything_fallback",
                default_value="",
                description="Override the Depth-Anything fallback flag from the config file.",
            ),
            DeclareLaunchArgument(
                "rosbag_path",
                default_value="",
                description="Override the rosbag path from the config file.",
            ),
            DeclareLaunchArgument(
                "rosbag_storage_id",
                default_value="",
                description="Override the rosbag storage id from the config file.",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
