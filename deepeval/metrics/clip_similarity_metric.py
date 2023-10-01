from deepeval.metrics.metric import Metric
from deepeval.test_case import ImageTestCase


class ClipSimilarityMetric:
    def __init__(self, model_name="ViT-B/32"):
        self.model_name = model_name
        import torch
        import clip

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def measure(self, test_case: ImageTestCase):
        import clip
        import torch
        from PIL import Image

        image_input = (
            self.preprocess(Image.open(test_case.image_path))
            .unsqueeze(0)
            .to(self.device)
        )
        text_inputs = torch.cat([clip.tokenize(test_case.query)]).to(
            self.device
        )
        # Calculate features
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_inputs)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return float(similarity[0])
