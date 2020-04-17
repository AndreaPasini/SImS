import networkx as nx

def json_to_nx(graph):
    """
    Convert json graph (node-link-data) to Networkx graph.
    """
    return nx.node_link_graph(graph, attrs = dict(source='s', target='r', name='id', key='key', link='links'))


def nx_to_json(graph):
    """
    Convert json graph (node-link-data) to Networkx graph.
    """
    return nx.node_link_data(graph, dict(source='s', target='r', name='id', key='key', link='links'))