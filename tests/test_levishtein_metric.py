from deepeval.metrics.levenshtein_distance_metric import LevenshteinDistanceMetric
def test_levishtein_metric():
    metric = LevenshteinDistanceMetric()
    result = metric.measure("This is weird", 'This is not weird')
    assert result == 3, result
