import numpy as np
from numpy.linalg import norm
from py_stringmatching import Jaccard, OverlapCoefficient, Cosine, TverskyIndex, Dice
from common.enums import SimilarityMethod


class Similarity:
    def __init__(self) -> None:
        pass

    def get_similarity(self, sim_method, a, b):
        if sim_method == SimilarityMethod.Jaccard:
            return self.jaccard(a, b)
        elif sim_method == SimilarityMethod.Cosine:
            return self.cosine(a, b)
        elif sim_method == SimilarityMethod.OverlapCoefficient:
            return self.overlap_coefficient(a, b)
        elif sim_method == SimilarityMethod.TverskyIndex:
            return self.tversky_index(a, b)
        elif sim_method == SimilarityMethod.Dice:
            return self.dice(a, b)
        else:
            raise Exception("unknown similarity method")

    def jaccard(self, a, b):
        method = Jaccard()
        return method.get_sim_score(a, b)

    def cosine(self, a, b):
        method = Cosine()
        return method.get_sim_score(a, b)

    def overlap_coefficient(self, a, b):
        method = OverlapCoefficient()
        return method.get_sim_score(a, b)

    def tversky_index(self, a, b):
        method = TverskyIndex()
        return method.get_sim_score(a, b)

    def dice(self, a, b):
        method = Dice()
        return method.get_sim_score(a, b)
