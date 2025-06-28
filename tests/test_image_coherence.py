from deepeval.metrics import (
    ImageCoherenceMetric,
    ImageHelpfulnessMetric,
    ImageReferenceMetric,
)
from deepeval.metrics.multimodal_metrics.multimodal_g_eval.multimodal_g_eval import (
    MultimodalGEval,
)
from deepeval.test_case import MLLMImage, MLLMTestCase
import textwrap
from deepeval import evaluate
from deepeval.test_case.mllm_test_case import MLLMTestCaseParams

online_url = "https://cdn.prod.website-files.com/64bd90bdba579d6cce245aec/6725d768a1a3be620ec4455c_1*IHcRH-dIpRpTp0edyq_9nA.png"
local_url = "./data/img.png"
image_url = online_url

mllm_param = [
    textwrap.dedent(
        """LLM Safety, a specialized area within AI Safety, focuses on safeguarding Large Language Models, ensuring they function responsibly and securely. This includes addressing vulnerabilities like data protection, content moderation, and reducing harmful or biased outputs in real-world applications.
        Government AI Regulations
        Just a few months ago, the European Union’s Artificial Intelligence Act (AI Act) came into force, marking the first-ever legal framework for AI. By setting common rules and regulations, the Act ensures that AI applications across the EU are safe, transparent, non-discriminatory, and environmentally sustainable.
    """
    ),
    MLLMImage(url=image_url),
    textwrap.dedent(
        """Alongside the EU’s AI Act, other countries are also advancing their efforts to improve safety standards and establish regulatory frameworks for AI and LLMs. These initiatives include:
        United States: AI Risk Management Framework by NIST (National Institute of Standards and Technology) and Executive Order 14110
        United Kingdom: Pro-Innovation AI Regulation by DSIT (Department for Science, Innovation and Technology)
        China: Generative AI Measures by CAC (Cyberspace Administration of China)
        Canada: Artificial Intelligence and Data Act (AIDA) by ISED (Innovation, Science, and Economic Development Canada)
        Japan: Draft AI Act by METI (Japan’s Ministry of Economy, Trade, and Industry)
        EU Artificial Intelligence Act (EU)
        The EU AI Act, which took effect in August 2024, provides a structured framework to ensure AI systems are used safely and responsibly across critical areas such as healthcare, public safety, education, and consumer protection.
    """
    ),
]

mllm_test_case = MLLMTestCase(
    input=mllm_param,
    actual_output=mllm_param,
    expected_output=mllm_param,
    retrieval_context=mllm_param,
    context=mllm_param,
)

###################################################
### Test evaluate #################################
###################################################

evaluate(
    test_cases=[mllm_test_case],
    metrics=[
        ImageCoherenceMetric(model="gpt-4.1"),
        MultimodalGEval(
            name="Answer Relevancy",
            model="gpt-4.1-nano",
            evaluation_params=[
                MLLMTestCaseParams.INPUT,
                MLLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            criteria="Determine if actual output is relevant to input.",
        ),
        # ImageCoherenceMetric(model="gpt-4.1"),
        # ImageCoherenceMetric(model="gpt-4.1"),
        # ImageCoherenceMetric(model="gpt-4.1"),
        # ImageCoherenceMetric(model="gpt-4.1"),
        # ImageCoherenceMetric(model="gpt-4.1"),
    ],
)


###################################################
### Test evaluate_image_coherence #################
###################################################

# image_coherence_metric = ImageCoherenceMetric()
# evaluation_1 = image_coherence_metric.evaluate_image_coherence(mllm_test_case.actual_output[1], mllm_test_case.actual_output[0], mllm_test_case.actual_output[2])
# evaluation_2 = image_coherence_metric.evaluate_image_coherence(mllm_test_case.actual_output[3], mllm_test_case.actual_output[2], mllm_test_case.actual_output[4])

# print(evaluation_1)
# print(evaluation_2)

###################################################
### Test measure ##################################
###################################################

# import time

# # Initialize metrics
# async_image_coherence_metric = ImageCoherenceMetric(
#     async_mode=True, verbose_mode=True
# )
# sync_image_coherence_metric = ImageCoherenceMetric(
#     async_mode=False, verbose_mode=True
# )

# # Measure time and evaluate async metric
# start_time_async = time.time()
# evaluation_1 = async_image_coherence_metric.measure(test_case=mllm_test_case)
# end_time_async = time.time()
# print(
#     f"Async Image Coherence Metric Evaluation Time: {end_time_async - start_time_async:.4f} seconds"
# )

# # Measure time and evaluate sync metric
# start_time_sync = time.time()
# evaluation_2 = sync_image_coherence_metric.measure(
#     test_case=mllm_test_case
# )  # Fixed typo here
# end_time_sync = time.time()
# print(
#     f"Sync Image Coherence Metric Evaluation Time: {end_time_sync - start_time_sync:.4f} seconds"
# )

# # Print evaluations
# print("Async Evaluation Result:", evaluation_1)
# print("Sync Evaluation Result:", evaluation_2)
