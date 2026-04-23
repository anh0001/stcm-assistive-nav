#!/usr/bin/env python3
"""Launch a local browser UI for STCM ground-truth annotation."""

import argparse
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'stcm')}")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "stcm"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from stcm.gt_annotation import (
    DEFAULT_BASE_FRAME,
    DEFAULT_CAMERA_INFO_TOPIC,
    DEFAULT_RGB_TOPIC,
    DEFAULT_TF_STATIC_TOPIC,
    DEFAULT_TF_TOPIC,
    DEFAULT_WORLD_FRAME,
    AnnotationSession,
    RosbagRgbIndex,
    default_output_json_path,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bag",
        default="/media/dl-box/STREAM1/ranger_recording_20251215_163827_uncompressed",
        help="Path to the rosbag2 directory for synchronized RGB/TF inspection.",
    )
    parser.add_argument(
        "--input-json",
        default=str(REPO_ROOT / "configs" / "experiments" / "ground_truth" / "meeting_stcm_gt.json"),
        help="Seed STCM JSON file to edit.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Draft output path. Defaults to <input>_draft.json beside the input JSON.",
    )
    parser.add_argument("--rgb-topic", default=DEFAULT_RGB_TOPIC)
    parser.add_argument("--camera-info-topic", default=DEFAULT_CAMERA_INFO_TOPIC)
    parser.add_argument("--tf-topic", default=DEFAULT_TF_TOPIC)
    parser.add_argument("--tf-static-topic", default=DEFAULT_TF_STATIC_TOPIC)
    parser.add_argument("--world-frame", default=DEFAULT_WORLD_FRAME)
    parser.add_argument("--base-frame", default=DEFAULT_BASE_FRAME)
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for the local Gradio server.")
    parser.add_argument("--port", type=int, default=7860, help="Port for the local Gradio server.")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Expose a Gradio share link. Off by default for local-only annotation.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "gradio is not installed. Install the Python requirements for STCM "
            "and rerun this script from the ROS-sourced environment."
        ) from exc

    args = _parse_args()
    input_json = Path(args.input_json).expanduser()
    output_json = Path(args.output_json).expanduser() if args.output_json else default_output_json_path(input_json)

    rosbag_index = RosbagRgbIndex(
        args.bag,
        rgb_topic=args.rgb_topic,
        camera_info_topic=args.camera_info_topic,
        tf_topic=args.tf_topic,
        tf_static_topic=args.tf_static_topic,
        world_frame=args.world_frame,
        base_frame=args.base_frame,
    )
    session = AnnotationSession(
        input_json=input_json,
        output_json=output_json,
        rosbag_index=rosbag_index,
    )

    sort_choices = ["id", "category", "x", "y", "z"]

    def _refresh(
        search_text: str,
        category_filter: str,
        sort_field: str,
        descending: bool,
        status_message: str = "",
    ):
        viewport = session.build_viewport()
        category_choices = ["All"] + session.list_categories()
        if category_filter not in category_choices:
            category_filter = "All"
        table_rows, table_ids = session.build_object_table(
            search_text=search_text,
            category_filter=category_filter,
            sort_field=sort_field,
            descending=descending,
        )
        selected_id, selected_category, pos_x, pos_y, pos_z = session.selected_object_fields()
        map_image = session.render_map(
            search_text=search_text,
            category_filter=category_filter,
            viewport=viewport,
        )
        rgb_image = session.selected_frame_image()
        return (
            map_image,
            rgb_image,
            table_rows,
            table_ids,
            selected_id,
            selected_category,
            pos_x,
            pos_y,
            pos_z,
            gr.update(choices=category_choices, value=category_filter),
            session.current_frame_index,
            status_message,
        )

    initial_category_filter = "All"
    initial_sort = "id"
    initial_desc = False
    (
        initial_map,
        initial_rgb,
        initial_table,
        initial_table_ids,
        initial_obj_id,
        initial_obj_category,
        initial_x,
        initial_y,
        initial_z,
        initial_filter_update,
        initial_frame_index,
        initial_status,
    ) = _refresh(
        "",
        initial_category_filter,
        initial_sort,
        initial_desc,
        status_message=(
            f"Loaded {len(session.objects)} semantic objects from {input_json.name}. "
            f"Draft saves will go to {output_json}."
        ),
    )

    with gr.Blocks(title="STCM GT Annotation") as demo:
        gr.Markdown("# STCM Ground-Truth Annotation")
        with gr.Row():
            with gr.Column(scale=5):
                map_image = gr.Image(
                    value=initial_map,
                    label="World View",
                    interactive=False,
                    type="numpy",
                    height=720,
                )
                frame_slider = gr.Slider(
                    minimum=0,
                    maximum=max(0, len(session.rosbag_index.frames) - 1),
                    step=1,
                    value=initial_frame_index,
                    label="RGB Frame Index",
                )
                mode_radio = gr.Radio(
                    choices=["inspect", "add", "move"],
                    value="inspect",
                    label="Map Click Mode",
                )
            with gr.Column(scale=4):
                rgb_image = gr.Image(
                    value=initial_rgb,
                    label="Synchronized RGB Frame",
                    interactive=False,
                    type="numpy",
                    height=420,
                )
                with gr.Row():
                    search_text = gr.Textbox(label="Filter Objects", placeholder="Search id or category")
                    category_filter = gr.Dropdown(
                        choices=initial_filter_update["choices"],
                        value=initial_filter_update["value"],
                        label="Category Filter",
                    )
                with gr.Row():
                    sort_field = gr.Dropdown(choices=sort_choices, value=initial_sort, label="Sort By")
                    sort_desc = gr.Checkbox(label="Descending", value=initial_desc)
                object_table = gr.Dataframe(
                    headers=["id", "category", "x", "y", "z"],
                    value=initial_table,
                    datatype=["str", "str", "number", "number", "number"],
                    label="Semantic Objects",
                    interactive=False,
                    row_count=(max(8, min(len(initial_table), 20)), "fixed"),
                    column_count=(5, "fixed"),
                )
                with gr.Row():
                    object_id = gr.Textbox(label="Object ID", value=initial_obj_id)
                    object_category = gr.Textbox(label="Category", value=initial_obj_category or "object")
                with gr.Row():
                    pos_x = gr.Number(label="X (m)", value=initial_x, precision=4)
                    pos_y = gr.Number(label="Y (m)", value=initial_y, precision=4)
                    pos_z = gr.Number(label="Z (m)", value=initial_z, precision=4)
                with gr.Row():
                    update_button = gr.Button("Apply Object Edit", variant="primary")
                    delete_button = gr.Button("Delete Selected")
                    confirm_delete = gr.Checkbox(label="Confirm delete", value=False)
                with gr.Row():
                    output_path = gr.Textbox(label="Draft Output Path", value=str(output_json))
                    save_button = gr.Button("Save Draft", variant="primary")
                status = gr.Markdown(initial_status)
                table_ids_state = gr.State(initial_table_ids)

        def refresh_from_controls(search_value, category_value, sort_value, desc_value):
            return _refresh(search_value, category_value, sort_value, desc_value, status_message="Updated view.")

        search_text.change(
            refresh_from_controls,
            inputs=[search_text, category_filter, sort_field, sort_desc],
            outputs=[
                map_image,
                rgb_image,
                object_table,
                table_ids_state,
                object_id,
                object_category,
                pos_x,
                pos_y,
                pos_z,
                category_filter,
                frame_slider,
                status,
            ],
        )
        category_filter.change(
            refresh_from_controls,
            inputs=[search_text, category_filter, sort_field, sort_desc],
            outputs=[
                map_image,
                rgb_image,
                object_table,
                table_ids_state,
                object_id,
                object_category,
                pos_x,
                pos_y,
                pos_z,
                category_filter,
                frame_slider,
                status,
            ],
        )
        sort_field.change(
            refresh_from_controls,
            inputs=[search_text, category_filter, sort_field, sort_desc],
            outputs=[
                map_image,
                rgb_image,
                object_table,
                table_ids_state,
                object_id,
                object_category,
                pos_x,
                pos_y,
                pos_z,
                category_filter,
                frame_slider,
                status,
            ],
        )
        sort_desc.change(
            refresh_from_controls,
            inputs=[search_text, category_filter, sort_field, sort_desc],
            outputs=[
                map_image,
                rgb_image,
                object_table,
                table_ids_state,
                object_id,
                object_category,
                pos_x,
                pos_y,
                pos_z,
                category_filter,
                frame_slider,
                status,
            ],
        )

        def on_frame_change(frame_value, search_value, category_value, sort_value, desc_value):
            message = session.set_frame_index(int(frame_value))
            return _refresh(search_value, category_value, sort_value, desc_value, status_message=message)

        frame_slider.change(
            on_frame_change,
            inputs=[frame_slider, search_text, category_filter, sort_field, sort_desc],
            outputs=[
                map_image,
                rgb_image,
                object_table,
                table_ids_state,
                object_id,
                object_category,
                pos_x,
                pos_y,
                pos_z,
                category_filter,
                frame_slider,
                status,
            ],
        )

        def on_table_select(evt: gr.SelectData, table_ids, search_value, category_value, sort_value, desc_value):
            index = getattr(evt, "index", None)
            if isinstance(index, (tuple, list)):
                row_index = int(index[0])
            else:
                row_index = int(index)
            message = session.select_table_row(row_index, list(table_ids or []))
            return _refresh(search_value, category_value, sort_value, desc_value, status_message=message)

        object_table.select(
            on_table_select,
            inputs=[table_ids_state, search_text, category_filter, sort_field, sort_desc],
            outputs=[
                map_image,
                rgb_image,
                object_table,
                table_ids_state,
                object_id,
                object_category,
                pos_x,
                pos_y,
                pos_z,
                category_filter,
                frame_slider,
                status,
            ],
        )

        def on_map_select(evt: gr.SelectData, mode_value, category_value, search_value, category_filter_value, sort_value, desc_value):
            index = getattr(evt, "index", None)
            if not isinstance(index, (tuple, list)) or len(index) < 2:
                message = "Map click did not return image coordinates."
            else:
                message = session.map_click(
                    int(index[0]),
                    int(index[1]),
                    mode=mode_value,
                    category_hint=category_value or "object",
                    viewport=session.build_viewport(),
                )
            return _refresh(search_value, category_filter_value, sort_value, desc_value, status_message=message)

        map_image.select(
            on_map_select,
            inputs=[mode_radio, object_category, search_text, category_filter, sort_field, sort_desc],
            outputs=[
                map_image,
                rgb_image,
                object_table,
                table_ids_state,
                object_id,
                object_category,
                pos_x,
                pos_y,
                pos_z,
                category_filter,
                frame_slider,
                status,
            ],
        )

        def on_update(obj_id_value, category_value, x_value, y_value, z_value, search_value, category_filter_value, sort_value, desc_value):
            message = session.update_selected_object(
                new_id=obj_id_value,
                category=category_value,
                x=float(x_value),
                y=float(y_value),
                z=float(z_value),
            )
            return _refresh(search_value, category_filter_value, sort_value, desc_value, status_message=message)

        update_button.click(
            on_update,
            inputs=[object_id, object_category, pos_x, pos_y, pos_z, search_text, category_filter, sort_field, sort_desc],
            outputs=[
                map_image,
                rgb_image,
                object_table,
                table_ids_state,
                object_id,
                object_category,
                pos_x,
                pos_y,
                pos_z,
                category_filter,
                frame_slider,
                status,
            ],
        )

        def on_delete(confirmed, search_value, category_filter_value, sort_value, desc_value):
            message = session.delete_selected_object(confirmed=bool(confirmed))
            refreshed = list(_refresh(search_value, category_filter_value, sort_value, desc_value, status_message=message))
            refreshed.append(False)
            return tuple(refreshed)

        delete_button.click(
            on_delete,
            inputs=[confirm_delete, search_text, category_filter, sort_field, sort_desc],
            outputs=[
                map_image,
                rgb_image,
                object_table,
                table_ids_state,
                object_id,
                object_category,
                pos_x,
                pos_y,
                pos_z,
                category_filter,
                frame_slider,
                status,
                confirm_delete,
            ],
        )

        def on_save(output_path_value):
            return session.save(output_path_value)

        save_button.click(on_save, inputs=[output_path], outputs=[status])

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
