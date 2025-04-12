import os

# Path to config
CURR_PATH = os.path.dirname(os.path.abspath(__file__))

# Set root path
ROOT_PATH = os.path.abspath(os.path.join(CURR_PATH, ".."))

# Data file path
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "profs_and_publications.json")

# Network edges path
EDGES_PATH = os.path.join(ROOT_PATH, "network", "coauthor_network_edge.csv")