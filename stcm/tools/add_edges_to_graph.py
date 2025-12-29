#!/usr/bin/env python3
"""
Utility script to retroactively add edges to an existing semantic graph.

This script reads a semantic graph JSON file (with no edges), calculates spatial
relationships between nodes based on distance, and adds edges to create a connected graph.
If the input is an STCM container, the semantic graph is updated in place and the
place graph is preserved.

Usage:
    python3 add_edges_to_graph.py <input_graph.json> [--output <output_graph.json>] [--distance <threshold>]

Examples:
    # Add edges to stcm.json with default 3.0m threshold
    python3 add_edges_to_graph.py stcm.json

    # Specify custom output file and distance threshold
    python3 add_edges_to_graph.py stcm.json --output stcm_with_edges.json --distance 2.5

    # Overwrite input file in-place
    python3 add_edges_to_graph.py stcm.json --output stcm.json
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import stcm modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from stcm.map_utils import read_stcm_json, save_graph_json, save_stcm_json, update_graph_edges


def main():
    parser = argparse.ArgumentParser(
        description="Add spatial relationship edges to an existing semantic graph JSON file"
    )
    parser.add_argument(
        "input_graph",
        type=str,
        help="Path to input semantic graph JSON file"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output graph JSON file (default: <input>_with_edges.json)"
    )
    parser.add_argument(
        "--distance",
        "-d",
        type=float,
        default=3.0,
        help="Maximum distance in meters to create edges between nodes (default: 3.0)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed information about edge creation"
    )

    args = parser.parse_args()

    input_path = Path(args.input_graph)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_with_edges{input_path.suffix}"

    # Load the graph
    print(f"Loading graph from: {input_path}")
    stcm_payload = read_stcm_json(str(input_path))
    graph = stcm_payload["semantic_graph"]

    num_nodes = graph.number_of_nodes()
    num_edges_before = graph.number_of_edges()

    print(f"Graph loaded: {num_nodes} nodes, {num_edges_before} edges")

    # Add edges based on spatial proximity
    print(f"Adding edges for nodes within {args.distance}m of each other...")
    update_graph_edges(graph, edge_distance_threshold=args.distance)

    num_edges_after = graph.number_of_edges()
    edges_added = num_edges_after - num_edges_before

    print(f"Edges added: {edges_added} (total edges: {num_edges_after})")

    if args.verbose and edges_added > 0:
        print("\nEdge details:")
        for u, v, data in graph.edges(data=True):
            u_category = graph.nodes[u].get('category', 'unknown')
            v_category = graph.nodes[v].get('category', 'unknown')
            distance = data.get('distance', 0.0)
            print(f"  {u} ({u_category}) <-> {v} ({v_category}): {distance:.2f}m")

    # Save the updated graph
    print(f"Saving graph to: {output_path}")
    if stcm_payload["is_stcm"]:
        save_stcm_json(
            graph,
            place_graph=stcm_payload["place_graph"],
            file=str(output_path),
            metadata=stcm_payload.get("metadata"),
        )
    else:
        save_graph_json(graph, file=str(output_path))

    print("Done!")


if __name__ == "__main__":
    main()
