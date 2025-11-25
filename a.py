from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import ToolCall

goldens = [
    Golden(
        input="Hi — I bought a blood pressure monitor last week for $55. Is this HSA-eligible under my employer plan? Can you show me the relevant policy text?",
        expected_output="Yes, blood pressure monitors are HSA-eligible. Excerpt: 'Durable medical equipment including blood pressure monitors is eligible for reimbursement.'",
        retrieval_context=[
            "Durable medical equipment including blood pressure monitors is eligible for reimbursement when used for the diagnosis or treatment of a medical condition.",
            "Examples of eligible HSA expenses: diagnostic devices (e.g., blood pressure monitors, glucose meters)."
        ],
        context=[
            "Durable medical equipment including blood pressure monitors is eligible for reimbursement when used for the diagnosis or treatment of a medical condition."
        ],
        tools_called=[
            ToolCall(name="auth_check", reasoning="Verify user identity", input_parameters={"user_id": "AUTO"}),
            ToolCall(name="plan_lookup", reasoning="Check HSA plan eligibility", input_parameters={"plan_id": "acme:hsa:A"})
        ],
        expected_tools=[
            ToolCall(name="auth_check", reasoning="Verify user identity", input_parameters={"user_id": "AUTO"}),
            ToolCall(name="kb_retrieval", reasoning="Fetch plan excerpt", input_parameters={"query": "blood pressure monitor eligible durable medical equipment", "top_k": 3})
        ]
    ),

    Golden(
        input="I uploaded my EOB from United Health. They applied my preventive screening to deductible — is that correct?",
        expected_output="Preventive screenings should be covered at 100% and not hit the deductible. Suggest creating an appeal with the EOB attached.",
        retrieval_context=[
            "Line: Preventive Screening — CPT 80050 — processed as subject to deductible.",
            "Preventive services covered at 100% — no deductible."
        ],
        context=[
            "Preventive services covered at 100% — no deductible."
        ],
        tools_called=[
            ToolCall(name="document_ingest", reasoning="Process uploaded EOB", input_parameters={"document_id": "uploaded:eob:united:claim12345"}),
            ToolCall(name="claim_analyzer", reasoning="Check claim processing", input_parameters={"claim_doc_id": "uploaded:eob:united:claim12345"})
        ],
        expected_tools=[
            ToolCall(name="kb_retrieval", reasoning="Retrieve plan coverage for preventive services", input_parameters={"query": "preventive services covered 100%"}),
            ToolCall(name="create_ticket", reasoning="Create appeal ticket", input_parameters={"type": "appeal"})
        ]
    ),

    Golden(
        input="Can you show me the investment options available for my HSA? I'm curious about risk levels and fees.",
        expected_output="Your HSA funds via Schwab: Vanguard Total Stock Market (High risk, 0.03%), Schwab Aggregate Bond (Moderate risk, 0.04%).",
        retrieval_context=[
            "User 123 linked funds: VTI, SCHZ. Expense ratios: 0.03%, 0.04%.",
            "Lively provides fund options via Charles Schwab; no personalized investment advice."
        ],
        context=[
            "User 123 linked funds: VTI, SCHZ."
        ],
        tools_called=[
            ToolCall(name="auth_check", reasoning="Verify user identity", input_parameters={"user_id": "AUTO"}),
            ToolCall(name="schwab_api_fetch", reasoning="Fetch linked fund info", input_parameters={"account_id": "user123"})
        ],
        expected_tools=[
            ToolCall(name="schwab_api_fetch", reasoning="Retrieve fund info", input_parameters={"account_id": "user123"}),
            ToolCall(name="response_formatter", reasoning="Format table of funds", input_parameters={"format": "table"})
        ]
    ),

    Golden(
        input="What's the diagnosis code on my EOB from Dr. Smith?",
        expected_output="Cannot provide PHI in chat; will open a secure message to support to get the diagnosis code.",
        retrieval_context=[
            "PHI must not be displayed in unencrypted chat.",
            "EOB uploaded by user contains diagnosis codes."
        ],
        context=[
            "PHI must not be displayed in unencrypted chat."
        ],
        tools_called=[
            ToolCall(name="auth_check", reasoning="Verify user identity", input_parameters={"user_id": "AUTO"}),
            ToolCall(name="secure_message_open", reasoning="Open secure flow for PHI", input_parameters={"recipient": "support"})
        ],
        expected_tools=[
            ToolCall(name="phi_policy_check", reasoning="Ensure PHI not exposed", input_parameters={"document_id": "uploaded:eob:drsmith:2025-06-10"}),
            ToolCall(name="secure_message_open", reasoning="Send PHI securely", input_parameters={"recipient": "support"})
        ]
    ),

    Golden(
        input="Can I use my HSA debit card to pay for a telehealth visit with HealthNet? Is that in-network or out-of-network?",
        expected_output="HSA card can be used; in-network HealthNet telehealth covered, out-of-network may apply to deductible.",
        retrieval_context=[
            "Telehealth visits with participating network providers are treated as in-network.",
            "Provider search endpoint: check provider network status."
        ],
        context=[
            "Telehealth visits with participating network providers are in-network."
        ],
        tools_called=[
            ToolCall(name="auth_check", reasoning="Verify user identity", input_parameters={"user_id": "AUTO"}),
            ToolCall(name="provider_lookup", reasoning="Check provider network status", input_parameters={"query": "HealthNet"})
        ],
        expected_tools=[
            ToolCall(name="plan_lookup", reasoning="Check telehealth coverage", input_parameters={"plan_id": "acme:hsa:A"}),
            ToolCall(name="provider_lookup", reasoning="Verify network status", input_parameters={"query": "HealthNet"})
        ]
    ),

    Golden(
        input="My $200 reimbursement for my diabetes supplies was denied. Why? What can I do?",
        expected_output="Denied due to insufficient documentation. Recommend uploading receipt with NPI and doctor's note for appeal.",
        retrieval_context=[
            "Denial reason: Not a covered expense — insufficient documentation.",
            "Documentation requirements: supplier NPI and proof of medical necessity."
        ],
        context=[
            "Documentation requirements: supplier NPI and proof of medical necessity."
        ],
        tools_called=[
            ToolCall(name="auth_check", reasoning="Verify user identity", input_parameters={"user_id": "AUTO"}),
            ToolCall(name="claim_analyzer", reasoning="Check denial reason", input_parameters={"claim_id": "7890"})
        ],
        expected_tools=[
            ToolCall(name="kb_retrieval", reasoning="Retrieve documentation requirements", input_parameters={"query": "HSA documentation requirements"}),
            ToolCall(name="create_ticket", reasoning="Create appeal ticket", input_parameters={"type": "appeal"})
        ]
    ),

    Golden(
        input="Between Plan A (low premium, high deductible) and Plan B (higher premium, low deductible), I expect 3 doctor visits and a $600 prescription this year. Which plan will likely cost me less over three months?",
        expected_output="Estimate: Plan A ≈ $750, Plan B ≈ $660. Based on this, Plan B may cost less for 3 months.",
        retrieval_context=[
            "Plan A: premium $50/mo, deductible $3000, office visit $40 after deductible.",
            "Plan B: premium $180/mo, deductible $500, office visit $20 copay."
        ],
        context=[
            "Plan A: premium $50/mo, deductible $3000.",
            "Plan B: premium $180/mo, deductible $500."
        ],
        tools_called=[
            ToolCall(name="auth_check", reasoning="Verify user identity", input_parameters={"user_id": "AUTO"}),
            ToolCall(name="cost_estimator", reasoning="Estimate plan costs", input_parameters={"usage": {"doctor_visits": 3, "prescription_cost": 600}})
        ],
        expected_tools=[
            ToolCall(name="kb_retrieval", reasoning="Retrieve plan details", input_parameters={"query": "Plan A Plan B premiums deductibles"}),
            ToolCall(name="cost_estimator", reasoning="Estimate plan costs", input_parameters={"usage": {"doctor_visits": 3, "prescription_cost": 600}})
        ]
    ),

    Golden(
        input="Is parking for my dialysis appointment reimbursable under my HSA?",
        expected_output="Yes, parking for dialysis is eligible. Excerpt: 'Qualified medical transportation, including parking fees for covered medical treatment, is eligible.'",
        retrieval_context=[
            "Parking fees for medical care are commonly reimbursable as qualified medical expenses.",
            "Qualified medical transportation, including parking fees for dialysis, is eligible for reimbursement."
        ],
        context=[
            "Qualified medical transportation, including parking fees for dialysis, is eligible."
        ],
        tools_called=[
            ToolCall(name="auth_check", reasoning="Verify user identity", input_parameters={"user_id": "AUTO"}),
            ToolCall(name="reimbursement_start", reasoning="Start reimbursement for parking", input_parameters={"receipt_id": "uploaded:receipt:parking:2025-06-02"})
        ],
        expected_tools=[
            ToolCall(name="kb_retrieval", reasoning="Retrieve eligibility info", input_parameters={"query": "parking dialysis eligible HSA"}),
            ToolCall(name="reimbursement_start", reasoning="Start reimbursement", input_parameters={"receipt_id": "uploaded:receipt:parking:2025-06-02"})
        ]
    ),

    Golden(
        input="I need to share my scanned invoice with HR but please remove my SSN and home address first.",
        expected_output="Redacted SSN and home address; preview ready for user approval.",
        retrieval_context=[
            "User-uploaded invoice containing SSN and home address.",
            "Redaction flow: identify SSN and addresses, replace with [REDACTED]."
        ],
        context=[
            "Redaction flow: identify SSN and addresses, replace with [REDACTED]."
        ],
        tools_called=[
            ToolCall(name="pii_redactor", reasoning="Redact sensitive info", input_parameters={"patterns": ["ssn","address"]}),
            ToolCall(name="activity_logger", reasoning="Log redaction action", input_parameters={"action": "redaction"})
        ],
        expected_tools=[
            ToolCall(name="pii_redactor", reasoning="Redact PII", input_parameters={"patterns": ["ssn","address"]}),
            ToolCall(name="response_formatter", reasoning="Preview redacted document", input_parameters={"format": "redaction_preview"})
        ]
    ),

    Golden(
        input="Can you schedule a 30-min benefits review with an HR rep and include an agenda covering HSA investment options and reimbursement timelines?",
        expected_output="Proposed 3 slots for 30-min meeting; will include agenda once slot confirmed.",
        retrieval_context=[
            "HR rep calendar free/busy: Tue 10:00, Wed 14:00, Thu 09:30 available.",
            "Use secure invites for sensitive benefits discussions; attach agenda."
        ],
        context=[
            "HR rep calendar free/busy: Tue 10:00, Wed 14:00, Thu 09:30 available."
        ],
        tools_called=[
            ToolCall(name="calendar_freebusy", reasoning="Check HR rep availability", input_parameters={"participant": "hr_rep_teamA"}),
            ToolCall(name="create_calendar_event", reasoning="Prepare event", input_parameters={"duration_minutes": 30})
        ],
        expected_tools=[
            ToolCall(name="create_calendar_event", reasoning="Create event after confirmation", input_parameters={"duration_minutes": 30}),
            ToolCall(name="secure_invite_sender", reasoning="Send secure invite", input_parameters={"recipients": ["user_email","hr_rep_email"]})
        ]
    )
]

dataset = EvaluationDataset(goldens=goldens)
dataset.push(alias="Single-turn Dataset")
