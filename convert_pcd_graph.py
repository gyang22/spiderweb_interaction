#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

def convert(nodes_path, edges_path, output_path):
    print(f"Loading nodes from: {nodes_path}")
    nodes = []
    with open(nodes_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            nodes.append([float(x) for x in parts])
    
    # Ensure 3D (Z might be 0)
    for i in range(len(nodes)):
        if len(nodes[i]) == 2:
            nodes[i].append(0.0)
    
    print(f"Loading edges from: {edges_path}")
    edges = []
    with open(edges_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                # First two columns are u, v
                edges.append([int(parts[0]), int(parts[1])])

    data = {
        "nodes": nodes,
        "edges": edges
    }

    print(f"Saving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Done! Created {output_path} with {len(nodes)} nodes and {len(edges)} edges.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PCD-Graph-Recon-DM text files to Spiderweb JSON format")
    parser.add_argument("nodes", help="Path to sorted-feature.txt")
    parser.add_argument("edges", help="Path to edge_detour_filtered.txt (or similar edge file)")
    parser.add_argument("output", help="Path to output JSON file")
    
    args = parser.parse_args()
    convert(args.nodes, args.edges, args.output)
