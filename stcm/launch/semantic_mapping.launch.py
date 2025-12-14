from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    text_prompt = LaunchConfiguration("text_prompt").perform(context)
    graph_path = LaunchConfiguration("graph_output_path").perform(context)
    run_updater = LaunchConfiguration("run_updater").perform(context).lower() in ("1", "true", "yes")

    builder_node = Node(
        package="stcm",
        executable="semantic_map_builder",
        name="semantic_map_builder",
        output="screen",
        parameters=[
            {"text_prompt": text_prompt},
            {"graph_output_path": graph_path},
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

    if run_updater:
        updater_node = Node(
            package="stcm",
            executable="semantic_map_updater",
            name="semantic_map_updater",
            output="screen",
            parameters=[
                {"text_prompt": text_prompt},
                {"graph_input_path": graph_path},
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ],
        )
        return [builder_node, updater_node]

    return [builder_node]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("text_prompt", default_value="table . door . chair ."),
            DeclareLaunchArgument("graph_output_path", default_value="graph.json"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("run_updater", default_value="false"),
            OpaqueFunction(function=launch_setup),
        ]
    )
