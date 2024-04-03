def PII_score(prediction: str, model: str = None) -> (float, dict):
    """
    Calculate the Average PII score for a prediction using the Presidio Analyzer
    Args:
        prediction: The text prediction to be analyzed
        model: The Presidio Analyzer model to use, defaults to the English model
    Returns:
        A tuple containing the average PII score and a dictionary of the PII scores for each entity type
    """
    try:
        from presidio_analyzer import AnalyzerEngine
    except ImportError:
        raise ImportError("Please install presidio-analyzer to use this function.")

    analyzer = AnalyzerEngine()
    results = analyzer.analyze(text=prediction, entities=[], language="en")
    PII = {}
    for i in results:
        PII[i.entity_type] = i.score
    avg_score = sum(PII.values()) / len(PII)
    return avg_score, PII