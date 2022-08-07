from operator import index
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Environment Path
main_data_dir = "D:\Github\HustMapSimpleNavigation\HustMap_Simple_Navigation\Graph"
adj_matrix = main_data_dir + "\Adjacency_Matrix_Preprocess.csv"
metadata = main_data_dir + "\Metadata.csv"

main_dir = ""
# DataFrame of ADJ and Metadata
meta_df = pd.read_csv(metadata)
adj_matrix_df = pd.read_csv(adj_matrix, header=None)
adj_matrix_df.columns = [x for x in range(35)]
print(adj_matrix_df.shape)
adj_shape = adj_matrix_df.shape

# Dictionary of Position of each node
position_df = meta_df[["Pos_X", "Pos_Y"]] / 500
position_dict = {x : (position_df.iloc[x, 0], position_df.iloc[x, 1]) for x in range(35)}

# Creating Graph
G = nx.Graph()
for i in range(adj_shape[0]):
    for j in range(adj_shape[1]):
        if adj_matrix_df.iloc[i, j] != 0:
            G.add_edge(i, j, weight = round(adj_matrix_df.iloc[i, j], 2))

nx.write_edgelist(G, path="grid.edgelist", delimiter=":")

elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.4]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.4]

nx.draw_networkx_nodes(G, position_dict, node_size=170)

nx.draw_networkx_edges(G, position_dict, edgelist=elarge, width=6)
nx.draw_networkx_edges(
    G, position_dict, 
    edgelist=esmall, 
    width=4, 
    alpha=0.5, 
    edge_color="b", 
    style="dashed")

nx.draw_networkx_labels(G, position_dict, font_size=8, font_family="sans-serif", font_color="w")

edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, position_dict, edge_labels, font_size=6)



# plt.figure(figsize=(40, 35))
ax = plt.gca()
ax.margins(0.08)
xmin, xmax, ymin, ymax = plt.axis([-1, 24, 18, 3])
print(xmin, " ", xmax, " ", ymin, " ", ymax)
plt.tight_layout()
plt.savefig("hustmap.png", format = 'png')
plt.show()