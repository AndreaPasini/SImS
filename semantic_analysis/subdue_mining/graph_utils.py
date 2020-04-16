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

def json_graph_to_subdue(graph, conv_coco_category, conv_pos_category):
    """
    Converts a json graph to Subdue format
    """
    vmap = {}  # old -> new
    vindex = 1  # The only difference with Gspan. Vertices start from 1

    descr = ""
    # for all nodes
    for v in graph['nodes']:
        l = conv_coco_category[v['label']]
        vmap[v['id']] = vindex
        descr += f"v {vindex} {l}\n"
        vindex += 1
    # for all edges
    for e in graph['links']:
        l = conv_pos_category[e['pos']]  # label
        s = vmap[e['s']]  # subject
        r = vmap[e['r']]  # reference
        descr += f"e {s} {r} {l}\n"

    return descr