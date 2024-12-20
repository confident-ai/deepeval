from deepeval.metrics import (
    ImageCoherenceMetric,
    ImageHelpfulnessMetric,
    ImageReferenceMetric,
)
from deepeval.test_case import MLLMImage, MLLMTestCase
import textwrap

mllm_test_case = MLLMTestCase(
    input=[],
    actual_output=[
        textwrap.dedent(
            """With great power comes great responsibility. As LLMs become more powerful, they are entrusted with increasing autonomy. This means less human oversight, greater access to personal data, and an ever-expanding role in handling real-life tasks.

            From managing weekly grocery orders to overseeing complex investment portfolios, LLMs present a tempting target for hackers and malicious actors eager to exploit them. Ignoring these risks could have serious ethical, legal, and financial repercussions. As pioneers of this technology, we have a duty to prioritize and uphold LLM safety.

            Although much of this territory is uncharted, it’s not entirely a black box. Governments worldwide are stepping up with new AI regulations, and extensive research is underway to develop risk mitigation strategies and frameworks. Today, we’ll dive into these topics, covering:

            What LLM Safety entails
            Government AI regulations and their impact on LLMs
            Key LLM vulnerabilities to watch out for
            Current LLM safety research, including essential risk mitigation strategies and frameworks
            Challenges in LLM safety and how Confident AI addresses these issues
            What is LLM Safety?
            LLM Safety combines practices, principles, and tools to ensure AI systems function as intended, focusing on aligning AI behavior with ethical standards to prevent unintended consequences and minimize harm.
        """
        ),
        MLLMImage(
            url="https://cdn.prod.website-files.com/64bd90bdba579d6cce245aec/6725d767c80255656d1f142c_1*VQIEl1L8JgolDQsqca-w4w.png"
        ),
        textwrap.dedent(
            """LLM Safety, a specialized area within AI Safety, focuses on safeguarding Large Language Models, ensuring they function responsibly and securely. This includes addressing vulnerabilities like data protection, content moderation, and reducing harmful or biased outputs in real-world applications.
            Government AI Regulations
            Just a few months ago, the European Union’s Artificial Intelligence Act (AI Act) came into force, marking the first-ever legal framework for AI. By setting common rules and regulations, the Act ensures that AI applications across the EU are safe, transparent, non-discriminatory, and environmentally sustainable.
        """
        ),
        MLLMImage(
            url="https://cdn.prod.website-files.com/64bd90bdba579d6cce245aec/6725d768a1a3be620ec4455c_1*IHcRH-dIpRpTp0edyq_9nA.png"
        ),
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
    * 2,
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

import time

# Initialize metrics
async_image_coherence_metric = ImageCoherenceMetric(
    async_mode=True, verbose_mode=True
)
sync_image_coherence_metric = ImageCoherenceMetric(
    async_mode=False, verbose_mode=True
)

# Measure time and evaluate async metric
start_time_async = time.time()
evaluation_1 = async_image_coherence_metric.measure(test_case=mllm_test_case)
end_time_async = time.time()
print(
    f"Async Image Coherence Metric Evaluation Time: {end_time_async - start_time_async:.4f} seconds"
)

# Measure time and evaluate sync metric
start_time_sync = time.time()
evaluation_2 = sync_image_coherence_metric.measure(
    test_case=mllm_test_case
)  # Fixed typo here
end_time_sync = time.time()
print(
    f"Sync Image Coherence Metric Evaluation Time: {end_time_sync - start_time_sync:.4f} seconds"
)

# Print evaluations
print("Async Evaluation Result:", evaluation_1)
print("Sync Evaluation Result:", evaluation_2)
