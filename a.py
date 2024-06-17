import asyncio
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

# Define the test cases
test_case1 = LLMTestCase(
    input="How is Artificial Intelligence (AI) being used to improve diagnostic accuracy in healthcare?",
    actual_output="""AI is revolutionizing the field of education by leveraging machine learning algorithms and deep learning techniques to analyze complex textbook data.""",
)
test_case2 = LLMTestCase(
    input="What ethical considerations arise with the implementation of AI in healthcare, and how can they be addressed?",
    actual_output="""The integration of AI in construction brings forth several ethical considerations, including data privacy, algorithmic bias, and the transparency of AI decision-making processes. Data privacy is a major concern, as AI systems often require access to large datasets of construction information, which must be handled with utmost care to prevent unauthorized access and breaches. Ensuring data anonymization and implementing robust security measures are essential steps to address this issue.""",
)
test_case3 = LLMTestCase(
    input="What role does AI play in personalized entertainment, and what are its potential benefits?",
    actual_output="""AI plays a pivotal role in advancing personalized medicine by enabling the analysis of large-scale datasets to uncover individual variations in patients' genetic profiles, lifestyle, and environmental factors. By integrating this diverse information, AI can identify unique biomarkers and predict responses to specific treatments, allowing for the development of personalized therapeutic strategies. One of the significant benefits of AI in personalized medicine is its ability to tailor treatments to an individual's genetic makeup, improving the efficacy of interventions and reducing adverse effects. For example, in oncology, AI algorithms can analyze tumor genomics to identify the most effective chemotherapy or targeted therapy for a particular patient, leading to better outcomes and fewer side effects. Additionally, AI-driven personalized medicine can facilitate early disease detection and prevention by identifying high-risk individuals and recommending proactive measures based on their unique health profiles. This approach not only enhances patient care but also has the potential to optimize healthcare resources by focusing on prevention and targeted treatments.""",
)
test_case4 = LLMTestCase(
    input="How can AI assist in streamlining repeatitive tasks in the automotive industry?",
    actual_output="""AI can significantly streamline administrative tasks in healthcare settings, reducing the burden on healthcare professionals and improving overall operational efficiency. One of the primary applications of AI in this domain is automating routine and repetitive tasks such as appointment scheduling, billing, and claims processing. AI-powered chatbots and virtual assistants can handle patient inquiries, schedule appointments, and send reminders, thereby freeing up staff time to focus on more complex tasks. Moreover, AI can enhance the accuracy and speed of medical billing and coding by extracting relevant information from clinical documents and ensuring compliance with coding standards, which reduces errors and accelerates the reimbursement process. AI can also aid in managing electronic health records (EHRs) by organizing and analyzing patient data, identifying patterns, and generating insights that support clinical decision-making. By automating these administrative processes, AI not only improves efficiency but also enhances patient satisfaction by reducing wait times and ensuring timely access to care.""",
)

# Define the metric
metric1 = AnswerRelevancyMetric(verbose_mode=True)
metric2 = AnswerRelevancyMetric(verbose_mode=True)

# metric1.measure(test_case1)
# metric2.measure(test_case2)


# Asynchronous function to measure relevancy concurrently
async def measure_relevancy():
    await asyncio.gather(
        metric1.a_measure(test_case1, _show_indicator=False),
        metric1.a_measure(test_case2, _show_indicator=False),
        metric1.a_measure(test_case3, _show_indicator=False),
        metric1.a_measure(test_case4, _show_indicator=False),
    )
    # await metric1.a_measure(test_case1, _show_indicator=False)
    print(metric1.statements)
    print("All measurements are done.")


# # Run the asynchronous function and print after completion
asyncio.run(measure_relevancy())
# print("This is printed after all asynchronous operations are complete.")


# print(metric1.statements)
# print(metric2.statements)
