from schemas.classifier import Classifier
from schemas.requests.common import ClassifierRequest

from ._linear import linear_classifier
from ._nearest_neighbors import nearest_neighbors

__all__ = ["nearest_neighbors", "linear_classifier"]


def get_classifier(request: ClassifierRequest):
    if request.classifier.name in [Classifier.EUCLIDEAN_NN, Classifier.COSINE_NN]:
        return nearest_neighbors
    elif request.classifier.name == Classifier.LINEAR:
        return linear_classifier
