# Ranking Similarity

If you are building retrieval-augmented generation applications, you may be constantly iterating on the embeddings andv ector search index.

Top-K rankings are evaluated with the following criteria:

- Top results are more important than bottom-ranked results.
  A drop from 2nd to 3rd is more significant than a drop from 5th to 6th and so on.
- A drop from 2nd to 3rd is more important than from 5th to 6th and so on.
- A specific result not appearing in another list is more indicative of difference than a result dropping in another list as it suggests the ranking itself is greater than K.

A specific result not appearing in another list is more indicative of difference than a result dropping in another list as it suggests the ranking itself is greater than K.

## Assert Ranking Similarity

In order to provide a suggestion on how to use ranking similarity, we are looking to build:

```python
from deepeval.test_utils import assert_ranking_similarity

result_list_1 = ["Sentence-1", "Sentence-2"]
result_list_2 = ["Sentence-2", "Sentence-3"]

assert_ranking_similarity(
    list1=result_list_1,
    list2=result_list_2,
    minimum_score=0.3
)
```

### Parameters

- `minimum_score` refers to the minimum score for this to be considered simiar ranking

## Ranking Similarity As A Metric

You can measure ranking similarity as a metric.

```python
from deepeval.metrics.ranking_similarity import RankingSimilarity
scorer = RankingSimilarity(minimum_score=0.5)
result = scorer.measure(list_1=list1, list_2=list2)
```

### Parameters

- `minimum_score` refers to the minimum score for this to be considered simiar ranking

### How it is measured

For ensuring top-k ranking similarity, we recommend the gentle introduction into the technique that is being used here.

https://medium.com/ai-in-plain-english/comparing-top-k-rankings-statistically-9adfc9cfc98b
