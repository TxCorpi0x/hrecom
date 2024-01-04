class Similarity:
    def __init__(self) -> None:
        pass

    def Jaccard(self, data, i, j):
        genres_i = data[i]
        genres_j = data[j]
        # print(genres_i)
        # print(genres_j)
        intersection_size = len(set(genres_i).intersection(genres_j))
        union_size = len(set(genres_i).union(genres_j))
        return intersection_size / union_size
