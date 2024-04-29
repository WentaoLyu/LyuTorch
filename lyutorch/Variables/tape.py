import networkx as nx


class Tape:
    def __init__(self) -> None:
        self.G = nx.DiGraph()
        self.graph_exists = False
        self.make_grad = True

    def add_edge(self, origin, destination):
        self.G.add_edge(origin, destination)

    def clear(self):
        self.G = nx.DiGraph()
        self.graph_exists = False

    def topo_sort(self):
        return list(nx.topological_sort(self.G))


tape = Tape()
