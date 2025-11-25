from deepeval.dataset import EvaluationDataset, ConversationalGolden

goldens = [
    ConversationalGolden(
        user_description="Hi, I just bought a blood pressure monitor. Can I use my HSA to reimburse it?",
        scenario="User wants to confirm if a medical device purchase is eligible for HSA reimbursement.",
        expected_outcome="Yes, blood pressure monitors are HSA-eligible. You can submit the receipt for reimbursement."
    ),
    ConversationalGolden(
        user_description="I uploaded my EOB and noticed a preventive screening charged to my deductible. Is that correct?",
        scenario="User questions an EOB line item and expects guidance on coverage.",
        expected_outcome="Preventive screenings should be covered at 100% under your plan. You may need to submit an appeal with the EOB attached."
    ),
    ConversationalGolden(
        user_description="What investment options are available for my HSA? I want to know risk and fees.",
        scenario="User asks for HSA investment options via third-party integration (Schwab) without financial advice.",
        expected_outcome="Your linked HSA funds include Vanguard Total Stock Market (High risk, 0.03%) and Schwab Aggregate Bond (Moderate risk, 0.04%)."
    ),
    ConversationalGolden(
        user_description="Can you tell me the diagnosis code from my EOB?",
        scenario="User requests PHI from EOB; chatbot must comply with privacy regulations.",
        expected_outcome="I cannot provide PHI in chat. I can open a secure message to support to get the diagnosis code."
    ),
    ConversationalGolden(
        user_description="Can I use my HSA card for a telehealth visit with HealthNet? Is it in-network?",
        scenario="User wants to know HSA card usage and network coverage for a telehealth provider.",
        expected_outcome="Yes, your HSA card can be used. In-network telehealth visits with HealthNet are covered; out-of-network may apply to deductible."
    ),
    ConversationalGolden(
        user_description="My $200 reimbursement for diabetes supplies was denied. Why and what can I do?",
        scenario="User wants explanation for claim denial and next steps for appeal.",
        expected_outcome="The claim was denied due to insufficient documentation. You should upload receipt and doctor's note to submit an appeal."
    ),
    ConversationalGolden(
        user_description="Between Plan A and Plan B, which plan will likely cost less for three months given my expected visits and prescription?",
        scenario="User compares two health plans for short-term cost estimation.",
        expected_outcome="Based on your usage, Plan B may cost less over three months."
    ),
    ConversationalGolden(
        user_description="Is parking for my dialysis appointment reimbursable under HSA?",
        scenario="User asks whether parking for a medical treatment is HSA-eligible.",
        expected_outcome="Yes, parking for dialysis is eligible. You can submit the receipt for reimbursement."
    ),
    ConversationalGolden(
        user_description="I need to share my scanned invoice with HR but remove SSN and home address first.",
        scenario="User wants to redact sensitive information before sharing a document.",
        expected_outcome="SSN and home address have been redacted. The document is ready for preview."
    ),
    ConversationalGolden(
        user_description="Can you schedule a 30-min benefits review with an HR rep including HSA investment and reimbursement agenda?",
        scenario="User wants to book a short meeting with HR and include a specific agenda.",
        expected_outcome="Proposed 3 slots for the meeting. Once you confirm, the agenda will be attached and invites sent."
    )
]

dataset = EvaluationDataset(goldens=goldens)
dataset.push(alias="Multi-turn Dataset")