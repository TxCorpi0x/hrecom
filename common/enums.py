from enum import Enum


class SimilarityMethod(Enum):
    Jaccard = 1
    Cosine = 2
    OverlapCoefficient = 3
    TverskyIndex = 4
    Dice = 5
