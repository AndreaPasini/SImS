import networkx as nx
import io
import matplotlib.image as mpimg
import graphviz

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

def nx_to_graphviz(graph):
    """
    Convert networkx graph (node-link-data) to Graphviz printable graph
    """
    g_viz = graphviz.Digraph()
    for n in graph.nodes(data=True):
        if 'label' in n:
            g_viz.node(str(n[0]), label=n['label'])
    for e in graph.edges(data=True):
        if 'pos' in e[2]:
            g_viz.edge(str(e[0]), str(e[1]), label=e[2]['pos'])
        elif 'rel' in e[2]:
            g_viz.edge(str(e[0]), str(e[1]), label=e[2]['rel'])
        else:
            g_viz.edge(str(e[0]), str(e[1]))
    g_viz.node_attr.update(style="filled", fillcolor='#e0f3db', fontsize="12")
    g_viz.edge_attr.update(fontsize="12")
    g_viz.graph_attr.update(dpi="150")
    return g_viz

def json_to_graphviz(graph):
    """
    Convert json graph (node-link-data) to Graphviz printable graph
    """
    g_viz = graphviz.Digraph()
    for n in graph['nodes']:
        g_viz.node(str(n['id']), label=n['label'])
    for l in graph['links']:
        g_viz.edge(str(l['s']), str(l['r']), label=l['pos'])
    g_viz.node_attr.update(style="filled", fillcolor='#e0f3db', fontsize="12")
    g_viz.edge_attr.update(fontsize="12")
    g_viz.graph_attr.update(dpi="150")
    return g_viz

def print_graph_picture(out_path, graph):
    """ Print networkx graph to file (picture). """
    A = nx.drawing.nx_agraph.to_agraph(graph)
    A.node_attr.update(style="filled", fillcolor='#e0f3db')
    A.layout('dot')
    A.draw(out_path)


def show_graphviz_graph(graph, ax):
    """
    :param graph: graphviz DiGraph
    :param ax: axs where graph should be printed
    """
    img2 = mpimg.imread(io.BytesIO(graph.pipe(format='png')),format='png')
    ax.imshow(img2,interpolation='spline16')

