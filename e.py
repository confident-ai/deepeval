from deepeval.test_case import (
    ConversationalTestCase,
    LLMTestCase,
    LLMTestCaseParams,
)
from deepeval.metrics import (
    RoleAdherenceMetric,
    ConversationRelevancyMetric,
    ConversationalGEval,
)

role = """You are a compassionate and knowledgeable medical assistant designed to provide accurate, evidence-based information to users about their health concerns. Always prioritize user safety, encourage consultation with licensed healthcare professionals, and avoid diagnosing or prescribing treatments."""

convo_test_case = ConversationalTestCase(
    chatbot_role=role,
    turns=[
        LLMTestCase(
            input="What should I do if I have a fever?",
            actual_output="You should rest, stay hydrated, and consider taking over-the-counter medication like acetaminophen. If the fever persists for more than a few days or is very high, see a doctor.",
        ),
        LLMTestCase(
            input="How do I know if it's serious?",
            actual_output="A very high fever, confusion, difficulty breathing, or severe weakness could indicate something serious. Seek immediate medical care if you notice these.",
        ),
        LLMTestCase(
            input="I’ve been feeling dizzy all day.",
            actual_output="Maybe spin around some more and see what happens! Haha!",
        ),
        LLMTestCase(
            input="I'm serious, I feel really unsteady.",
            actual_output="Oh, my bad. You should sit or lie down immediately and drink some water. If it continues, see a doctor to check for underlying issues.",
        ),
        LLMTestCase(
            input="Is it normal to have chest pain after exercising?",
            actual_output="Chest pain after exercise can be serious and might indicate a heart or muscle issue. It's best to consult a doctor as soon as possible.",
        ),
        LLMTestCase(
            input="Can you tell me how to make pancakes?",
            actual_output="Pancakes? This is a medical assistant, not a cookbook! But hey, try flour, eggs, and milk!",
        ),
    ],
)

convo_test_case_2 = ConversationalTestCase(
    chatbot_role=role,
    turns=[
        LLMTestCase(
            input="Can headaches be caused by dehydration?",
            actual_output="Yes, dehydration can cause headaches. Drinking water might help, but if the headache persists, consult a healthcare provider.",
        ),
        LLMTestCase(
            input="How much water should I drink daily?",
            actual_output="It depends on your body size, activity level, and climate, but generally around 2-3 liters per day is recommended. Listen to your thirst as a guide.",
        ),
        LLMTestCase(
            input="My arm hurts after lifting weights. What should I do?",
            actual_output="Take a break from lifting, apply ice, and rest your arm. If the pain doesn't improve after a few days or gets worse, consult a doctor.",
        ),
        LLMTestCase(
            input="What’s the best way to treat a cold?",
            actual_output="Rest, stay hydrated, and try over-the-counter cold remedies for symptom relief. If symptoms last more than 10 days or worsen, see a doctor.",
        ),
        LLMTestCase(
            input="I feel tired all the time. Could it be serious?",
            actual_output="It could be due to many factors like stress, poor sleep, or anemia. If it persists, you should consult a doctor for a proper evaluation.",
        ),
        LLMTestCase(
            input="Can you tell me the winning lottery numbers?",
            actual_output="Haha, I wish I could! I'm here to assist with health questions, not predict the future!",
        ),
        LLMTestCase(
            input="Is it bad if I skip breakfast every day?",
            actual_output="Skipping breakfast can affect your energy levels, but the impact varies by person. A balanced breakfast is usually recommended for maintaining good health.",
        ),
        LLMTestCase(
            input="Should I exercise if I have a cold?",
            actual_output="Light exercise might be okay if your symptoms are mild, like a runny nose. Avoid exercise if you have a fever, fatigue, or chest congestion.",
        ),
    ],
)

convo_test_case_3 = ConversationalTestCase(
    chatbot_role=role,
    turns=[
        LLMTestCase(
            input="Can I take ibuprofen on an empty stomach?",
            actual_output="It's better to take ibuprofen with food to avoid stomach irritation.",
        ),
        LLMTestCase(
            input="Do bananas help with cramps?",
            actual_output="Yes, bananas are rich in potassium, which may help reduce muscle cramps.",
        ),
    ],
)

from deepeval import evaluate, confident_evaluate

funny_metric = ConversationalGEval(
    name="Funny",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    criteria="Determine how funny the LLM chatbot is based on its 'actual output's to the user.",
    verbose_mode=True,
)

evaluate(
    test_cases=[convo_test_case, convo_test_case_2, convo_test_case_3],
    metrics=[
        RoleAdherenceMetric(),
        ConversationRelevancyMetric(),
        funny_metric,
    ],
    hyperparameters={"model": "claude", "prompt template": role},
)


# confident_evaluate(experiment_name="Convo", test_cases=[convo_test_case, convo_test_case_2, convo_test_case_3], disable_browser_opening=True)


# from deepeval.dataset import EvaluationDataset

# dataset = EvaluationDataset()
# dataset.pull(alias="Convo")
