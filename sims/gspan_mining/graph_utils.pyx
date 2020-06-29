import networkx as nx


def json_graph_to_gspan(graph, conv_coco_category, conv_pos_category):
    """
    Converts a json graph to Gspan format
    """
    vmap = {}  # old -> new
    vindex = 0

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
        if s!=r: # Self loop not allowed in gSpan
            descr += f"e {s} {r} {l}\n"
        else:
            print("Warning: found self loop in graphs.")

    return descr


