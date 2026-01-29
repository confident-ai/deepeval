"""
tests/test_integrations/test_openai_agents/apps/handoff_chain_app.py
Tests multi-hop agent handoffs (Triage -> Support -> Billing).
"""

from deepeval.openai_agents import Agent

def get_handoff_chain_app():
    # 3. Leaf Agent
    billing_agent = Agent(
        name="Billing Specialist",
        instructions="You handle refunds. Reply 'Refund Processed' to any request.",
    )

    # 2. Middle Agent
    support_agent = Agent(
        name="Support Agent",
        instructions="You are general support. If the user asks for a refund, handoff to the Billing Specialist immediately.",
        handoffs=[billing_agent],
    )

    # 1. Root Agent
    triage_agent = Agent(
        name="Triage Bot",
        instructions="You are the front desk. If the user has a complaint, handoff to the Support Agent.",
        handoffs=[support_agent],
    )

    return triage_agent, "I have a complaint and I want a refund."