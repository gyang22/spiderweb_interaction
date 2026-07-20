# GUI PCD interaction client

## Requirements
Use **Python 3.11** — `open3d` does not ship wheels for 3.12+, so newer
interpreters will fail to install the dependencies.

```bash
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run
run python3 main.py

# json_graphs branch (merged)
First, create a graph using PCD Graph Recon DM.
Then convert the graph to json format.
```python
# 1. Convert the graph
python3 convert_pcd_graph.py graph/sorted-feature.txt graph/edge_detour_filtered.txt graph/my_graph.json

# 2. View in the UI
python3 main.py
# (In the app, use File > Import Skeleton JSON...)

```
