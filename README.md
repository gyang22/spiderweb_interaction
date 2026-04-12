# GUI PCD interaction client

run python3 main.py

# json_graphs branch
First, create a graph using PCD Graph Recon DM.
Then convert the graph to json format.
```python
# 1. Convert the graph
python3 convert_pcd_graph.py graph/sorted-feature.txt graph/edge_detour_filtered.txt graph/my_graph.json

# 2. View in the UI
python3 main.py
# (In the app, use File > Import Skeleton JSON...)

```
