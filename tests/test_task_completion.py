from deepeval.metrics import (
    ToolCorrectnessMetric,
    AnswerRelevancyMetric,
    TaskCompletionMetric,
)
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval import evaluate, assert_test

test_case_travel_itinerary = LLMTestCase(
    input="Plan a 3-day itinerary for a trip to Paris, including cultural landmarks and local cuisine recommendations.",
    actual_output="Day 1: Visit the Eiffel Tower, have dinner at Le Jules Verne. Day 2: Explore the Louvre Museum, lunch at Angelina Paris. Day 3: Walk through Montmartre, evening at a wine bar.",
    tools_called=[
        ToolCall(
            name="Itinerary Generator",
            description="Generates travel itineraries based on destination and duration.",
            input_parameters={"destination": "Paris", "days": 3},
            output=[
                "Day 1: Eiffel Tower, Le Jules Verne.",
                "Day 2: Louvre Museum, Angelina Paris.",
                "Day 3: Montmartre, wine bar.",
            ],
        ),
        ToolCall(
            name="Restaurant Finder",
            description="Finds top restaurants in a given city.",
            input_parameters={"city": "Paris"},
            output=["Le Jules Verne", "Angelina Paris", "local wine bars"],
        ),
    ],
)

test_case_job_interview = LLMTestCase(
    input="I have a job interview for a software engineering role. Provide 5 key questions I should prepare for and tips on answering them.",
    actual_output="1. 'Tell me about yourself.' Focus on your relevant skills. 2. 'Describe a challenging project.' Highlight problem-solving. 3. 'Why this company?' Research their values. 4. 'What are your strengths?' Be specific. 5. 'Do you have questions for us?' Ask about growth opportunities.",
    tools_called=[
        ToolCall(
            name="Question Generator",
            description="Generates interview questions for a specific job role.",
            input_parameters={"role": "software engineer"},
            output=[
                "Tell me about yourself.",
                "Describe a challenging project you've worked on.",
                "Why do you want to work here?",
                "What are your strengths?",
                "Do you have questions for us?",
            ],
        ),
        ToolCall(
            name="Answer Tips Provider",
            description="Provides advice on how to answer specific interview questions.",
            input_parameters={
                "questions": [
                    "Tell me about yourself.",
                    "Describe a challenging project.",
                ]
            },
            output=[
                "Focus on relevant skills and achievements.",
                "Highlight problem-solving and technical expertise.",
            ],
        ),
    ],
)

test_case_email_drafting = LLMTestCase(
    input="Write a professional email to follow up after a job interview for a marketing role.",
    actual_output="Subject: Thank You for the Opportunity\nDear [Interviewer Name],\nThank you for taking the time to meet with me about the marketing role. I’m excited about the opportunity and look forward to contributing to your team.\nBest regards,\n[Your Name]",
    tools_called=[
        ToolCall(
            name="Email Template Generator",
            description="Generates email templates based on the given purpose and tone.",
            input_parameters={"purpose": "follow-up", "role": "marketing"},
            output="Subject: Thank You for the Opportunity\nDear [Interviewer Name],\nThank you for taking the time to meet with me about the marketing role. I’m excited about the opportunity and look forward to contributing to your team.\nBest regards,\n[Your Name]",
        ),
    ],
)

test_case_recipe_recommendation = LLMTestCase(
    input="I have chicken, spinach, and garlic. Suggest a recipe.",
    actual_output="You can make garlic butter chicken with sautéed spinach. Ingredients: chicken, garlic, spinach, butter, salt, and pepper. Instructions: Sauté garlic, cook chicken in butter, and add spinach.",
    tools_called=[
        ToolCall(
            name="Recipe Finder",
            description="Finds recipes based on the provided ingredients.",
            input_parameters={"ingredients": ["chicken", "spinach", "garlic"]},
            output="Garlic butter chicken with sautéed spinach. Ingredients: chicken, garlic, spinach, butter, salt, and pepper.",
        ),
        ToolCall(
            name="Cooking Steps Generator",
            description="Provides step-by-step instructions for a given recipe.",
            input_parameters={
                "recipe": "Garlic butter chicken with sautéed spinach"
            },
            output="1. Sauté garlic. 2. Cook chicken in butter. 3. Add spinach and season with salt and pepper.",
        ),
    ],
)

test_case_meeting_scheduler = LLMTestCase(
    input="Schedule a 30-minute meeting with the team next week. Preferred times are between 10 AM and 12 PM.",
    actual_output="The meeting is scheduled for Wednesday at 10:30 AM.",
    tools_called=[
        ToolCall(
            name="Calendar Scheduler",
            description="Schedules meetings based on availability and preferences.",
            input_parameters={
                "duration": "30 minutes",
                "time_slots": ["10:00 AM - 12:00 PM"],
                "participants": "team",
                "date_range": "next week",
            },
            output="Wednesday at 10:30 AM",
        ),
    ],
)

test_case_vague_input = LLMTestCase(
    input="Tell me something interesting.",
    actual_output="The Eiffel Tower is in Paris.",
    tools_called=[
        ToolCall(
            name="Fact Generator",
            description="Generates random facts.",
            input_parameters={},
            output="The Eiffel Tower is in Paris.",
        ),
    ],
)

test_case_overlapping_tools = LLMTestCase(
    input="Book a dinner reservation for 2 at 7 PM tomorrow at an Italian restaurant.",
    actual_output="Reservation confirmed at La Piazza at 7 PM.",
    tools_called=[
        ToolCall(
            name="Restaurant Finder",
            description="Finds Italian restaurants nearby.",
            input_parameters={"cuisine": "Italian"},
            output=["La Piazza", "Olive Garden"],
        ),
        ToolCall(
            name="Reservation Scheduler",
            description="Schedules reservations.",
            input_parameters={"restaurant": "La Piazza", "time": "7 PM"},
            output="Reservation confirmed at La Piazza at 7 PM.",
        ),
        ToolCall(
            name="Weather Checker",
            description="Checks the weather.",
            input_parameters={"time": "7 PM", "location": "unknown"},
            output="Clear skies.",
        ),
    ],
)

test_case_contradictory_output = LLMTestCase(
    input="Summarize the article about climate change in 50 words.",
    actual_output="Climate change is causing global warming and affecting ecosystems. We must reduce emissions and adopt renewable energy sources. Many countries are taking steps, but more effort is needed to avoid catastrophic impacts. Collaboration is crucial.",
    tools_called=[
        ToolCall(
            name="Article Summarizer",
            description="Summarizes articles to a specified word count.",
            input_parameters={
                "article": "climate change article",
                "word_count": 50,
            },
            output="Climate change is causing global warming and affecting ecosystems.",
        ),
    ],
)

test_case_missing_inputs = LLMTestCase(
    input="Translate 'Bonjour' to English.",
    actual_output="Hello.",
    tools_called=[
        ToolCall(
            name="Translation Tool",
            description="Translates text from one language to another.",
            input_parameters={},
            output="Hello.",
        ),
    ],
)

test_case_unrealistic_expectations = LLMTestCase(
    input="Write a 500-page novel about space exploration.",
    actual_output="The story is about a brave astronaut exploring Mars and discovering alien life.",
    tools_called=[
        ToolCall(
            name="Novel Generator",
            description="Generates stories based on a theme.",
            input_parameters={
                "theme": "space exploration",
                "length": "500 pages",
            },
            output="The story is about a brave astronaut exploring Mars.",
        ),
    ],
)

test_case_ambiguous_output = LLMTestCase(
    input="What’s the capital of the USA?",
    actual_output="It’s either Washington, D.C. or New York.",
    tools_called=[
        ToolCall(
            name="Fact Checker",
            description="Retrieves factual information.",
            input_parameters={"query": "capital of the USA"},
            output="It’s either Washington, D.C. or New York.",
        ),
    ],
)

# List of bad test cases
bad_test_cases = [
    test_case_vague_input,
    test_case_overlapping_tools,
    test_case_contradictory_output,
    test_case_missing_inputs,
    test_case_unrealistic_expectations,
    test_case_ambiguous_output,
]

# List of good test cases
test_cases = [
    test_case_travel_itinerary,
    test_case_job_interview,
    test_case_email_drafting,
    test_case_recipe_recommendation,
    test_case_meeting_scheduler,
]

task_completion_metric = TaskCompletionMetric()
evaluate(
    test_cases=test_cases + bad_test_cases,
    metrics=[task_completion_metric],
    verbose_mode=True,
    run_async=False,
)
