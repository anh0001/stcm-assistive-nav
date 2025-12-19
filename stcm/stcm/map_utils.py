import numpy as np
import os
import cv2
import yaml
from numpy.linalg import norm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
import json

intrinsics = [
    [574.0527954101562, 0.0, 319.5],
    [0.0, 574.0527954101562, 239.5],
    [0.0, 0.0, 1.0],
]
# intrinsics = [[554.254691191187, 0.0, 320.5],[0.0, 554.254691191187, 240.5], [0.0, 0.0, 1.0]]
fx = intrinsics[0][0]
fy = intrinsics[1][1]
px = intrinsics[0][2]
py = intrinsics[1][2]


def _intrinsics_values(override=None):
    if override is None:
        return fx, fy, px, py
    return (
        override.get("fx", fx),
        override.get("fy", fy),
        override.get("px", px),
        override.get("py", py),
    )


def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1)  # Shape: [H x W x 3]
    return xyz_img


def _cloud_field_names(cloud_array):
    if isinstance(cloud_array, dict):
        return set(cloud_array.keys())
    return set(cloud_array.dtype.names or ())


def _extract_cloud_xyz(cloud_array):
    field_names = _cloud_field_names(cloud_array)
    if {"x", "y", "z"}.issubset(field_names):
        points = np.stack((cloud_array["x"], cloud_array["y"], cloud_array["z"]), axis=1)
        return points.astype(np.float32, copy=False)
    if {"x_lidar", "y_lidar", "z_lidar"}.issubset(field_names):
        points = np.stack(
            (cloud_array["x_lidar"], cloud_array["y_lidar"], cloud_array["z_lidar"]),
            axis=1,
        )
        return points.astype(np.float32, copy=False)
    return None


def pose_in_map_frame_from_projected(
    projected_cloud,
    rt_cloud,
    rt_base,
    segment=None,
    mask_threshold=0.5,
    rt_camera=None,
):
    if projected_cloud is None or rt_cloud is None or rt_base is None:
        return None

    field_names = _cloud_field_names(projected_cloud)
    if "u" not in field_names or "v" not in field_names:
        return None

    xyz = _extract_cloud_xyz(projected_cloud)
    if xyz is None:
        return None

    u_coords = np.asarray(projected_cloud["u"], dtype=np.float32)
    v_coords = np.asarray(projected_cloud["v"], dtype=np.float32)

    valid = np.isfinite(u_coords) & np.isfinite(v_coords)
    if xyz.shape[0] != u_coords.shape[0]:
        return None
    valid &= np.isfinite(xyz).all(axis=1)
    valid &= ~(np.all(xyz == 0.0, axis=1))
    if not np.any(valid):
        return None

    xyz = xyz[valid]
    u_coords = u_coords[valid]
    v_coords = v_coords[valid]

    u_idx = np.rint(u_coords).astype(np.int32)
    v_idx = np.rint(v_coords).astype(np.int32)

    if segment is not None:
        height, width = segment.shape[:2]
        inside = (u_idx >= 0) & (u_idx < width) & (v_idx >= 0) & (v_idx < height)
        if not np.any(inside):
            return None
        xyz = xyz[inside]
        u_idx = u_idx[inside]
        v_idx = v_idx[inside]

        mask = segment[v_idx, u_idx] > mask_threshold
        if not np.any(mask):
            return None
        xyz = xyz[mask]
        u_idx = u_idx[mask]
        v_idx = v_idx[mask]
    elif xyz.size == 0:
        return None

    if xyz.size == 0:
        return None

    width = int(np.max(u_idx)) + 1 if segment is None else width
    linear_idx = v_idx * width + u_idx
    if rt_camera is not None:
        rt_camera_inv = np.linalg.inv(rt_camera)
        rt_cloud_to_camera = rt_camera_inv @ rt_cloud
        xyz_camera = np.dot(rt_cloud_to_camera[:3, :3], xyz.T).T + rt_cloud_to_camera[:3, 3]
        depth_vals = xyz_camera[:, 2]
        depth_valid = np.isfinite(depth_vals) & (depth_vals > 0.0)
        if not np.any(depth_valid):
            return None
        xyz = xyz[depth_valid]
        linear_idx = linear_idx[depth_valid]
        depth_vals = depth_vals[depth_valid]
    else:
        depth_vals = np.linalg.norm(xyz, axis=1)

    order = np.lexsort((depth_vals, linear_idx))
    unique_pixels, first_idx = np.unique(linear_idx[order], return_index=True)
    if unique_pixels.size == 0:
        return None
    keep = order[first_idx]
    xyz = xyz[keep]

    # Transform: lidar → base_link → map
    xyz_base = np.dot(rt_cloud[:3, :3], xyz.T).T
    xyz_base += rt_cloud[:3, 3]

    xyz_map = np.dot(rt_base[:3, :3], xyz_base.T).T
    xyz_map += rt_base[:3, 3]

    mean_pose = np.mean(xyz_map, axis=0)
    return mean_pose.tolist()


def pose_to_map_pixel(map_metadata, pose):
    pose_x = pose[0]
    pose_y = pose[1]

    map_pixel_x = int((pose_x - map_metadata["origin"][0]) / map_metadata["resolution"])
    map_pixel_y = int((pose_y - map_metadata["origin"][1]) / map_metadata["resolution"])

    return [map_pixel_x, map_pixel_y]


def pose_along_line(pose1, pose2, distance=2):
    '''
    creates a new pose that is at the specified distance from pose1
    along the line from pose1 to pose2
    '''
    pose2 = pose2[0:3,3]
    difference_vector = pose2 - pose1
    unit_vector = difference_vector / norm(difference_vector)
    new_pose = pose1 + unit_vector * distance

    return new_pose


def read_map_image(map_file_path):
    assert os.path.exists(map_file_path)
    if map_file_path.endswith(".pgm"):
        map_image = cv2.imread(map_file_path)
    else:
        map_image = cv2.imread(map_file_path)

    return map_image


def read_map_metadata(metadata_file_path):
    assert os.path.exists(metadata_file_path)
    assert metadata_file_path.endswith(".yaml")
    with open(metadata_file_path, "r") as file:
        metadata = yaml.safe_load(file)
    file.close()
    return metadata


def display_map_image(map_image, write=False):
    width, height, _ = map_image.shape
    cv2.namedWindow("Map Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Map Image", width, height)
    if write:
        cv2.imwrite("map_image.png", map_image)
    cv2.imshow("Map Image", map_image)
    cv2.waitKey(0)


def is_nearby(pose1, pose2, threshold=0.5):
    if norm((pose1[0] - pose2[0], pose1[1] - pose2[1])) < threshold:
        return True


def normalize_depth_image(depth_array, max_depth):
    depth_image = (max_depth - depth_array) / max_depth
    depth_image = depth_image * 255
    return depth_image.astype(np.uint8)


def denormalize_depth_image(depth_image, max_depth):

    depth_array = max_depth * (1 - (depth_image / 255))
    # print(f"max {depth_array.max()}")
    return depth_array.astype(np.float32)

def get_fov_points_in_baselink(depth_array, RT_camera, intrinsics=None):
        fx_v, fy_v, px_v, py_v = _intrinsics_values(intrinsics)
        mask1 = np.isnan(depth_array)
        depth_array[mask1] = 0.0
        xyz_array = compute_xyz(
            depth_array, fx_v, fy_v, px_v, py_v, depth_array.shape[0], depth_array.shape[1]
        )
        xyz_array = xyz_array.reshape((-1, 3))

        mask = ~(np.all(xyz_array == [0.0, 0.0, 0.0], axis=1))
        xyz_array = xyz_array[mask]

        xyz_base = np.dot(RT_camera[:3, :3], xyz_array.T).T
        xyz_base += RT_camera[:3, 3]

        min_x = np.min(xyz_base[:,0])
        max_x = np.max(xyz_base[:,0])
        min_y = np.min(xyz_base[:,1])
        max_y = np.max(xyz_base[:,1])

        return [[0,0,0],[max_x,min_y,0], [max_x, max_y,0]]

def get_fov_points_in_map(depth_array, RT_camera, RT_base, intrinsics=None):
        fx_v, fy_v, px_v, py_v = _intrinsics_values(intrinsics)
        mask1 = np.isnan(depth_array)
        depth_array[mask1] = 0.0
        xyz_array = compute_xyz(
            depth_array, fx_v, fy_v, px_v, py_v, depth_array.shape[0], depth_array.shape[1]
        )
        xyz_array = xyz_array.reshape((-1, 3))

        mask = ~(np.all(xyz_array == [0.0, 0.0, 0.0], axis=1))
        xyz_array = xyz_array[mask]

        xyz_base = np.dot(RT_camera[:3, :3], xyz_array.T).T
        xyz_base += RT_camera[:3, 3]

        min_x = np.min(xyz_base[:,0])
        max_x = np.max(xyz_base[:,0])
        min_y = np.min(xyz_base[:,1])
        max_y = np.max(xyz_base[:,1])

        points_baselink = [[0,0,0],[max_x,min_y,0], [max_x, max_y,0]]
        points_map = np.dot(RT_base[:3,:3], np.array(points_baselink).T).T + RT_base[:3,3]

        return points_map.tolist()

def pose_in_map_frame(RT_camera, RT_base, depth_array, segment=None, intrinsics=None):
    fx_v, fy_v, px_v, py_v = _intrinsics_values(intrinsics)
    if segment is not None:
        depth_array = depth_array * (segment / 1)

    #TODO: if depth is not normalized, then we need to remoev nans in the read image 
    # depth_array[np.isnan(depth_array)] = 0.0
    mask1 = np.isnan(depth_array)
    depth_array[mask1] = 0.0
    

    if depth_array.max() == 0.0:
        return None
    else:
        xyz_array = compute_xyz(
            depth_array, fx_v, fy_v, px_v, py_v, depth_array.shape[0], depth_array.shape[1]
        )
        xyz_array = xyz_array.reshape((-1, 3))

        mask = ~(np.all(xyz_array == [0.0, 0.0, 0.0], axis=1))
        xyz_array = xyz_array[mask]
        xyz_base = np.dot(RT_camera[:3, :3], xyz_array.T).T
        xyz_base += RT_camera[:3, 3]

        xyz_map = np.dot(RT_base[:3, :3], xyz_base.T).T
        xyz_map += RT_base[:3, 3]

        mean_pose = np.mean(xyz_map, axis=0)
        # mean_pose = pose_along_line( mean_pose, RT_base)
        return mean_pose.tolist()


def is_nearby_in_map(pose_list, node_pose, threshold=0.5):
    if len(pose_list) == 0:
        return pose_list, False
    pose_array = np.array(pose_list)
    node_pose_array = np.array([node_pose])
    distances = np.linalg.norm((pose_array[:, 0:2] - node_pose_array[:, 0:2]), axis=1)
    if np.any(distances < threshold):
        # print("not a new object")
        return pose_list, True
    else:
        # print("new node added")
        pose_list.append(node_pose)
        # print(f"pose list after {pose_list}")
        return pose_list, False


def update_graph_edges(graph, edge_distance_threshold=3.0):
    """
    Create edges between nodes based on spatial proximity.

    Args:
        graph: NetworkX graph with nodes containing 'pose' attribute
        edge_distance_threshold: Maximum distance in meters to create an edge between nodes

    Returns:
        Updated graph with edges representing spatial relationships
    """
    nodes_list = list(graph.nodes(data=True))

    for i, (node1_id, node1_data) in enumerate(nodes_list):
        pose1 = node1_data.get("pose")
        if pose1 is None or len(pose1) < 2:
            continue
        pose1 = np.array(pose1[:2], dtype=float)

        for j in range(i + 1, len(nodes_list)):
            node2_id, node2_data = nodes_list[j]
            pose2 = node2_data.get("pose")
            if pose2 is None or len(pose2) < 2:
                continue
            pose2 = np.array(pose2[:2], dtype=float)

            distance = np.linalg.norm(pose1 - pose2)

            if distance <= edge_distance_threshold:
                if not graph.has_edge(node1_id, node2_id):
                    graph.add_edge(node1_id, node2_id, distance=float(distance))

    return graph


def save_graph_json(graph, file="graph.json"):
    '''
    input graph \n
    save graph to graph.json
    '''
    file = file
    data_to_save = json_graph.node_link_data(graph)
    with open(file, "w") as file:
        json.dump(data_to_save, file, indent=4)
        file.close()


def read_graph_json(file="graph.json"):
    with open(file, "r") as file:
        data = json.load(file)
        file.close()
    # print(data)
    graph = json_graph.node_link_graph(data)
    return graph


def read_and_visualize_graph(map_file_path, map_metadata_filepath, on_map=False, catgeories=[], graph=None):
    if graph is None:
        graph = read_graph_json()
    else:
        graph = graph
    color_palette = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    if not on_map:
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True)
        plt.show()
    else:
        # c ncj
        map_image = read_map_image(map_file_path)
        map_metadata = read_map_metadata(map_metadata_filepath)
        for node, data in graph.nodes(data=True):
            if data["category"] in catgeories:
                x, y = pose_to_map_pixel(map_metadata, data["pose"])
                map_image[
                    y - 10 // 2 : y + 10 // 2,
                    x - 10 // 2 : x + 10 // 2,
                    :,
                ] = color_palette[catgeories.index(data["category"])]
        display_map_image(map_image, write=True)

def plot_point_on_map(map_file_path, map_metadata_filepath, position):
    map_image = read_map_image(map_file_path)
    map_metadata = read_map_metadata(map_metadata_filepath)
    x, y = pose_to_map_pixel(map_metadata, position)
    map_image[
        y - 10 // 2 : y + 10 // 2,
        x - 10 // 2 : x + 10 // 2,
        :,
    ] = [0,0,255]
    display_map_image(map_image, write=False)

def visualize_graph(graph):
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True)
    plt.show()


if __name__ == "__main__":
    # a=compute_xyz(np.array([[0,0,0],[0,0,0],[0,0,0]]), fx,fy,px,py, 3,3)
    # a=a.reshape((-1,3))
    # print(a)
    # save_graph_json()
    graph = read_graph_json()
    read_and_visualize_graph("map.png","map.yaml", on_map=True, catgeories=["table", "chair", "door"], graph=graph)
