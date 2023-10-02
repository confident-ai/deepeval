# Image Similarity

Image similarity provides a simple way to compare images that if it fails, it will retry.


## Text And Image Similarity

Here, you can provide an analysis of text and image similarity.

```python
from deepeval.metrics.clip_similarity_metric import ClipSimilarityMetric
from deepeval.test_case import ImageTestCase

# Show-casing 
test_case = ImageTestCase(
    image_path="photo_1.jpg",
    query="San Francisco"
)
metric = ClipSimilarityMetric()
score = metric.measure(test_case)
```

## Image and Image Similarity

You can use image to image similarity with the following.

```python
from deepeval.metrics.clip_similarity_metric import ClipSimilarityMetric
from deepeval.test_case import ImageTestCase

# Show-casing 
test_case = ImageTestCase(
    image_path="photo_1.jpg",
    ground_truth_image_path="photo_2.jpg",
)
metric = ClipSimilarityMetric()
score = metric.measure(test_case)
```

### Under the hood

Under the hood, it uses CLIP to measure cosine similarity between images and text and outputs the cosine similarity score.

## Parameters

This image similarity introduces the `ImageTestCase` that compares 2 images and text with image. The image test case has the following parameters:

- `image_path` - The path of the generated image
- `query` - The specified query
- `ground_truth_image_path` - The ground truth image path mentions that 
