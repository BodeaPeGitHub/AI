def greedy_communities_detection_by_tool(network):
    from networkx.algorithms import community
    import numpy as np
    import networkx as nx

    numpy_matrix = np.matrix(network["mat"])
    networkx_graf = nx.from_numpy_matrix(numpy_matrix)
    communities_generator = community.girvan_newman(networkx_graf)
    top_level_communities = next(communities_generator)
    sorted(map(sorted, top_level_communities))
    communities = [0] * network['noNodes']
    index = 1
    for community in sorted(map(sorted, top_level_communities)):
        for node in community:
            communities[node] = index
        index += 1
    return communities
