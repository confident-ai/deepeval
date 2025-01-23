from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

# See above for contents of fake data
fake_data = [
    {
        "input": "I have a persistent cough and fever. Should I be worried?",
        "actual_output": (
            "Based on your symptoms, it could be a sign of a viral or bacterial infection. "
            "However, if the fever persists for more than three days or you experience difficulty breathing, "
            "please consult a doctor immediately."
        ),
        "retrieval_context": [
            "Coughing that lasts more than three weeks is typically classified as a chronic cough and could indicate conditions such as asthma, chronic bronchitis, or gastroesophageal reflux disease (GERD).",
            "A fever is the body's natural response to infections, often caused by viruses or bacteria. Persistent fevers lasting more than three days should be evaluated by a healthcare professional as they may indicate conditions like pneumonia, tuberculosis, or sepsis.",
            "Shortness of breath associated with fever and cough can be a sign of serious respiratory issues such as pneumonia, bronchitis, or COVID-19.",
            "Self-care tips for mild symptoms include staying hydrated, taking over-the-counter fever reducers (e.g., acetaminophen or ibuprofen), and resting. Avoid suppressing a productive cough without consulting a healthcare provider.",
        ],
    },
    {
        "input": "What should I do if I accidentally cut my finger deeply?",
        "actual_output": (
            "If you cut your finger deeply, just rinse it with water and avoid applying any pressure. "
            "Tetanus shots aren't necessary unless you see redness immediately."
        ),
        "retrieval_context": [
            "Deep cuts that are more than 0.25 inches deep or expose fat, muscle, or bone require immediate medical attention. Such wounds may need stitches to heal properly.",
            "To minimize the risk of infection, wash the wound thoroughly with soap and water. Avoid using alcohol or hydrogen peroxide, as these can irritate the tissue and delay healing.",
            "If the bleeding persists for more than 10 minutes or soaks through multiple layers of cloth or bandages, seek emergency care. Continuous bleeding might indicate damage to an artery or vein.",
            "Watch for signs of infection, including redness, swelling, warmth, pain, or pus. Infections can develop even in small cuts if not properly cleaned or if the individual is at risk (e.g., diabetic or immunocompromised).",
            "Tetanus, a bacterial infection caused by Clostridium tetani, can enter the body through open wounds. Ensure that your tetanus vaccination is up to date, especially if the wound was caused by a rusty or dirty object.",
        ],
    },
]


# Create a list of LLMTestCase
test_cases = []
for fake_datum in fake_data:
    test_case = LLMTestCase(
        input=fake_datum["input"],
        actual_output=fake_datum["actual_output"],
        retrieval_context=fake_datum["retrieval_context"],
    )
    test_cases.append(test_case)

# Define metrics
answer_relevancy = AnswerRelevancyMetric()
faithfulness = FaithfulnessMetric()

# Run evaluation
evaluate(test_cases=test_cases, metrics=[answer_relevancy, faithfulness])
