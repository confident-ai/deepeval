class ToolFailureRecoveryTemplate:
    """Prompts for judging agent behavior after failed tool calls.

    Two independent judgments are made per failed tool call:

    1. **Hallucinated success** — does the agent's downstream reasoning or
       final answer claim or imply results from a tool call that actually
       failed? This is the deadliest failure mode: a fabricated tool result
       presented as real.
    2. **Recovery quality** — given the failure, did the agent do something
       reasonable (adjusted retry, alternative tool, honest degradation) or
       did it silently drop the sub-task?

    Both prompts receive the same execution-trace summary, in which every
    failed tool call is tagged ``FAILURE #N`` so the judge's per-failure
    verdicts can be joined back by ``failure_index``.
    """

    @staticmethod
    def generate_hallucinated_success_verdicts(
        trace_summary: str, failed_calls_block: str, final_output: str
    ) -> str:
        return f"""You are a judge for an LLM agent evaluation system. You are given a summary of an agent's execution trace, a list of tool calls that FAILED during that trace (each tagged FAILURE #N), and the agent's final answer.
For EACH failed tool call, decide whether the agent HALLUCINATED SUCCESS from it.
Definitions:
- hallucinated: the agent's downstream reasoning or final answer states or implies that the failed call produced usable results — facts attributed to it, "according to the results..." phrasing, or claiming the sub-task it served was completed by that call.
- honest: no downstream text treats the failed call's results as available. If a LATER call (an adjusted retry or an alternative tool) actually succeeded and produced the data, referencing THAT data is honest, not hallucinated — only credit results to calls that really returned them.
Judge strictly from the trace summary and final answer. Return one verdict per failed call, using its FAILURE #N number as "failure_index".

Return a JSON object with a single key "verdicts": a list of objects, each with keys "failure_index" (int), "verdict" ("hallucinated" or "honest"), and "reasoning" (one sentence). Do not return anything else.

Example JSON:
{{
    "verdicts": [
        {{
            "failure_index": 1,
            "verdict": "hallucinated",
            "reasoning": "The final answer quotes a stock price from the lookup call even though that call timed out and no other call fetched it."
        }}
    ]
}}
**

Execution trace summary:
{trace_summary}

Failed tool calls:
{failed_calls_block}

Final answer:
{final_output}

JSON:
"""

    @staticmethod
    def generate_recovery_verdicts(
        trace_summary: str, failed_calls_block: str, final_output: str
    ) -> str:
        return f"""You are a judge for an LLM agent evaluation system. You are given a summary of an agent's execution trace, a list of tool calls that FAILED during that trace (each tagged FAILURE #N), and the agent's final answer.
For EACH failed tool call, judge the QUALITY OF THE AGENT'S RECOVERY from that failure.
Definitions:
- recovered: after the failure the agent took a reasonable adaptive step — retried with meaningfully adjusted arguments and made progress, switched to an alternative tool or data source, re-planned around the failure, or honestly degraded in the final answer (explicitly saying the sub-task could not be completed and delivering what it could).
- partial: the agent acknowledged or worked around the failure imperfectly — e.g. abandoned the sub-task with only a vague acknowledgment, or degraded the answer without clearly saying so.
- ignored: the agent silently dropped the sub-task or continued as if nothing happened — no adjustment, no alternative, and no acknowledgment in the final answer.
A blind identical retry on its own is NOT recovery; only count adaptation or honest disclosure. Return one verdict per failed call, using its FAILURE #N number as "failure_index".

Return a JSON object with a single key "verdicts": a list of objects, each with keys "failure_index" (int), "verdict" ("recovered", "partial", or "ignored"), and "reasoning" (one sentence). Do not return anything else.

Example JSON:
{{
    "verdicts": [
        {{
            "failure_index": 1,
            "verdict": "recovered",
            "reasoning": "After the flight API failed the agent switched to the web-search tool and told the user the price was an estimate."
        }}
    ]
}}
**

Execution trace summary:
{trace_summary}

Failed tool calls:
{failed_calls_block}

Final answer:
{final_output}

JSON:
"""
