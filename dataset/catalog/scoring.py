from dataset.metadata.models import DatasetMetadata, QualityRating


class ScoringCriterion:
    """
    Base class for a dataset scoring criterion.
    """

    def __init__(self, name: str, weight: float):
        self.name = name
        self.weight = weight

    def evaluate(self, metadata: DatasetMetadata) -> float:
        """
        Evaluate the metadata and return a normalized score between 0.0 and 1.0.
        """
        raise NotImplementedError


class QualityScoringFramework:
    """
    Framework for evaluating and scoring datasets.
    Calculates a weighted average score based on registered criteria.
    """

    def __init__(self):
        self.criteria: dict[str, ScoringCriterion] = {}

    def add_criterion(self, criterion: ScoringCriterion):
        self.criteria[criterion.name] = criterion

    def score_dataset(self, metadata: DatasetMetadata) -> dict[str, float]:
        """
        Returns a dictionary with individual criterion scores and the 'total_score'.
        """
        if not self.criteria:
            return {"total_score": 0.0}

        total_weight = sum(c.weight for c in self.criteria.values())
        if total_weight == 0:
            return {"total_score": 0.0}

        scores = {}
        weighted_sum = 0.0

        for name, criterion in self.criteria.items():
            score = criterion.evaluate(metadata)
            scores[name] = score
            weighted_sum += score * criterion.weight

        scores["total_score"] = weighted_sum / total_weight
        return scores

    @staticmethod
    def map_score_to_rating(score: float) -> QualityRating:
        if score >= 0.8:
            return QualityRating.HIGH
        elif score >= 0.5:
            return QualityRating.MEDIUM
        return QualityRating.LOW
