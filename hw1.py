def manhattan_dist(r1, r2):
    """ Arguments r1 and r2 are lists of numbers """
    raise NotImplementedError()


def euclidean_dist(r1, r2):
    raise NotImplementedError()


def single_linkage(c1, c2, distance_fn):
    """ Arguments c1 and c2 are lists of lists of numbers
    (lists of input vectors or rows).
    Argument distance_fn is a function that can compute
    a distance between two vectors (like manhattan_dist)."""
    raise NotImplementedError()


def complete_linkage(c1, c2, distance_fn):
    raise NotImplementedError()


def average_linkage(c1, c2, distance_fn):
    raise NotImplementedError()


class HierarchicalClustering:

    def __init__(self, cluster_dist, return_distances=False):
        # the function that measures distances clusters (lists of data vectors)
        self.cluster_dist = cluster_dist

        # if the results of run() also needs to include distances;
        # if true, each joined pair in also described by a distance.
        self.return_distances = return_distances

    def closest_clusters(self, data, clusters):
        """
        Return the closest pair of clusters and their distance.
        """
        raise NotImplementedError()

    def run(self, data):
        """
        Performs hierarchical clustering until there is only a single cluster left
        and return a recursive structure of clusters.
        """

        # clusters stores current clustering. It starts as a list of lists
        # of single elements, but then evolves into lists like
        # [[["Albert"], [["Branka"], ["Cene"]]], [["Nika"], ["Polona"]]]
        clusters = [[name] for name in data.keys()]

        while len(clusters) >= 2:
            first, second, distance = self.closest_clusters(data, clusters)
            # update the "clusters" variable
            raise NotImplementedError()

        return clusters


if __name__ == "__main__":

    data = {"a": [1, 2],
            "b": [2, 3],
            "c": [5, 5]}

    def average_linkage_w_manhattan(c1, c2):
        return average_linkage(c1, c2, manhattan_dist)

    hc = HierarchicalClustering(cluster_dist=average_linkage_w_manhattan)
    clusters = hc.run(data)
    print(clusters)  # [[['c'], [['a'], ['b']]]] (or equivalent)

    hc = HierarchicalClustering(cluster_dist=average_linkage_w_manhattan,
                                return_distances=True)
    clusters = hc.run(data)
    print(clusters)  # [[['c'], [['a'], ['b'], 2.0], 6.0]] (or equivalent)
