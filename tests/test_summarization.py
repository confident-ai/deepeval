import pytest
from deepeval import assert_test
from deepeval.metrics.summarization.schema import Verdicts
from deepeval.test_case import LLMTestCase
from deepeval.metrics import SummarizationMetric
from tests.custom_judge import CustomJudge


@pytest.mark.skip(reason="openai isg expensive")
def test_summarization():
    metric = SummarizationMetric(verbose_mode=True, truths_extraction_limit=2)

    input = """
    In the rapidly evolving digital landscape, the proliferation of artificial intelligence (AI) technologies has been a game-changer in various industries, ranging from healthcare to finance. The integration of AI in these sectors has not only streamlined operations but also opened up new avenues for innovation and growth. In healthcare, AI algorithms are increasingly being used for diagnostic purposes, analyzing medical images, and providing personalized medicine solutions. This has significantly improved patient outcomes and has the potential to revolutionize healthcare delivery systems globally. For example, AI-driven tools can now detect anomalies in medical images with greater accuracy and speed than traditional methods, aiding in early diagnosis and treatment of diseases like cancer.

    In the financial sector, AI has transformed how businesses manage their finances and interact with customers. Automated trading systems, powered by AI algorithms, can analyze market trends and execute trades at speeds unattainable by human traders. This has led to increased efficiency and profitability for financial institutions. Additionally, AI is being used in fraud detection, where its ability to analyze vast amounts of transaction data quickly helps in identifying fraudulent activities, thereby protecting both the institutions and their customers.

    Moreover, AI's impact extends beyond these industries. In the field of autonomous vehicles, AI is the cornerstone technology enabling self-driving cars to make split-second decisions based on real-time data, enhancing road safety. The technology's learning algorithms can adapt to new driving conditions and environments, further pushing the boundaries of innovation in transportation.

    The ethical implications of AI are equally noteworthy. As AI systems become more advanced, concerns about privacy, data security, and the potential for AI-driven automation to impact employment have come to the forefront. It is crucial for policymakers and industry leaders to address these challenges and establish robust frameworks to ensure that AI technologies are developed and utilized responsibly and ethically.

    In conclusion, the impact of AI across various sectors is profound and multifaceted. Its ability to process and analyze data rapidly, coupled with its adaptability and precision, positions AI as a transformative force in the modern world. However, it is imperative to navigate the challenges it presents with foresight and responsibility, to fully harness its potential for the betterment of society.
    """

    output = """
    Artificial Intelligence (AI) is significantly influencing numerous industries, notably healthcare, finance, and education. In healthcare, AI aids in diagnostics and personalized medicine, improving patient outcomes through efficient analysis of medical data. In finance, AI enhances market trend analysis and fraud detection, boosting efficiency and security. AI's role in developing autonomous vehicles is also pivotal, promoting road safety through real-time decision-making. However, the rapid growth of AI raises ethical concerns, including privacy, data security, and employment impact. Addressing these issues is vital for the responsible development and use of AI, which holds great potential for societal advancement. Also, OpenAI is the leader in the AI race.
    """

    test_case = LLMTestCase(input=input, actual_output=output)

    assert_test(test_case, [metric])


def test_verdict_schema():

    judge = CustomJudge("mock")
    schema = Verdicts
    answer = (
        '{\n"verdicts": [\n{\n"verdict": "yes"\n},\n{\n    "verdict": "no",\n    "reason": "blah blah"\n},'
        '\n{\n    "verdict": "yes",\n    "reason":null \n}\n]\n}'
    )
    res: Verdicts = judge.generate(answer, schema=schema)
