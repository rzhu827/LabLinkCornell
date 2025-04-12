import os

# Set root path
ROOT_PATH = os.path.abspath(os.path.join("..", os.curdir))

# Data file path
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "profs_and_publications.json")

# Network edges path
EDGES_PATH = os.path.join(ROOT_PATH, "network", "coauthor_network_edge.csv")