import math

def manhattan_dist(r1, r2):
    """ Arguments r1 and r2 are lists of numbers """
    sum = 0
    leftout = 0
    diff = 0
    for i in range(0, len(r1)):
        if(math.isnan(r1[i]) or math.isnan(r2[i])):
            leftout += 1
        else:
            diff += abs(r1[i] - r2[i])
            sum += abs(r1[i] - r2[i])
    
    if(leftout == len(r1)):
        return math.nan

    sum += diff/(len(r1) - leftout) * leftout
    return sum 


def euclidean_dist(r1, r2):
    leftout = 0
    sum = 0
    diff = 0
    for i in range(0, len(r1)):
        if(math.isnan(r1[i]) or math.isnan(r2[i])):
            leftout += 1
        else:
            diff += abs(r1[i] - r2[i])
            sum += abs(math.pow(r1[i] - r2[i], 2))

    if(leftout == len(r1)):
        return math.nan

    sum += diff/(len(r1) - leftout) * leftout
    return math.sqrt(sum)


def single_linkage(c1, c2, distance_fn):
    """ Arguments c1 and c2 are lists of lists of numbers
    (lists of input vectors or rows).
    Argument distance_fn is a function that can compute
    a distance between two vectors (like manhattan_dist)."""
    min = math.inf
    for i1 in c1:
        for i2 in c2:
            dist = distance_fn(i1, i2)
            if((not math.isnan(dist)) and dist < min):
                min = dist

    if (min == math.inf):
        return math.nan

    return min


def complete_linkage(c1, c2, distance_fn):
    max = -math.inf
    for i1 in c1:
        for i2 in c2:
            dist = distance_fn(i1, i2)
            if((not math.isnan(dist)) and dist > max):
                max = dist

    if (max == -math.inf):
        return math.nan

    return max


def average_linkage(c1, c2, distance_fn):
    sum = 0
    used = 0
    for i1 in c1:
        for i2 in c2:
            dist = distance_fn(i1, i2)
            if (not math.isnan(dist)):
                used += 1
                sum += dist

    if(used == 0):
        return math.nan
    
    return sum/(used)


class HierarchicalClustering:

    def __init__(self, cluster_dist, return_distances=False):
        # the function that measures distances clusters (lists of data vectors)
        self.cluster_dist = cluster_dist

        # if the results of run() also needs to include distances;
        # if true, each joined pair in also described by a distance.
        self.return_distances = return_distances
   
    def get_cluster(self, c):
        cluster = []
        if(len(c) == 1):
            return c
        self.cluster_rec(c, cluster)
        return cluster

    def cluster_rec(self, c, cluster):
        for i in c:
            if(len(i) == 1):
                cluster.append(i[0])
            else:
                self.cluster_rec(i, cluster)

    def norm_clusters(self, clusters):
        cnorm =  []
        for c in clusters:
            cluster = self.get_cluster(c)
            cnorm.append(cluster)
            #print(cluster)
        return cnorm

    def closest_clusters(self, data, clusters):
        """
        Return the closest pair of clusters and their distance.
        """
        cnorm = self.norm_clusters(clusters)
        clusters_data = []

        for c in cnorm:
            #print(c)
            cd = []
            for lbl in c:
                cd.append(data[lbl])
            #print(cd)
            clusters_data.append(cd)
        
        closest = [0,0]
        min = math.inf

        for i1 in range(0,len(clusters_data)):
            for i2 in range(0,len(clusters_data)):
                if (i1 == i2):
                    continue
                dist = self.cluster_dist(clusters_data[i1], clusters_data[i2])
                if ((not math.isnan(dist)) and dist < min):
                    closest[0] = i1
                    closest[1] = i2
                    min = dist
        
        print(closest, min)

        return clusters[closest[0]], clusters[closest[1]], min

         
    def run(self, data):
        """
        Performs hierarchical clustering until there is only a single cluster left
        and return a recursive structure of clusters.
        """

        # clusters stores current clustering. It starts as a list of lists
        # of single elements, but then evolves into lists like
        # [[["Albert"], [["Branka"], ["Cene"]]], [["Nika"], ["Polona"]]]
        clusters = [[name] for name in data.keys()]
        ret = [[name] for name in data.keys()]

        while len(clusters) >= 2:
            first, second, distance = self.closest_clusters(data, clusters)
            f = clusters.index(first)
            s = clusters.index(second)
            if self.return_distances:
                mrt = [ret[f], ret[s], distance]
            else:    
                mrt = [ret[f], ret[s]]

            merged = [clusters[f], clusters[s]]
            clusters[f] = merged
            clusters.pop(s)
            #print(clusters)

            ret[f]= mrt
            ret.pop(s)
            print(ret)

            #raise NotImplementedError()

        return ret


if __name__ == "__main__":

    data = {"a": [1, 2],
            "b": [2, 3],
            "c": [5, 5]}

    #print(data["c"])

    def average_linkage_w_manhattan(c1, c2):
        return average_linkage(c1, c2, manhattan_dist)

    hc = HierarchicalClustering(cluster_dist=average_linkage_w_manhattan)
    clusters = hc.run(data)
    print(clusters)  # [[['c'], [['a'], ['b']]]] (or equivalent)

    hc = HierarchicalClustering(cluster_dist=average_linkage_w_manhattan,
                                return_distances=True)
    clusters = hc.run(data)
    print(clusters)  # [[['c'], [['a'], ['b'], 2.0], 6.0]] (or equivalent)
