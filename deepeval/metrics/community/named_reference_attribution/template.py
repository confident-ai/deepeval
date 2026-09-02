from typing import List

from deepeval.metrics.community.named_reference_attribution.schema import (
    NamedReference,
)


class NamedReferenceAttributionTemplate:
    """Prompts for named-reference attribution.

    Unlike `CitationFaithfulnessMetric` (which checks whether `[N]` citation
    markers assigned by the metric itself point to a supporting passage), this
    checks references to a document's *own* structural labels: "Table 3",
    "Section 4.2", "footnote 4", and similar names that exist in the source
    document independent of how a RAG pipeline chunked it. It is designed to
    catch a model citing the right kind of fact under the wrong named label,
    e.g. answering "Table 3" with a figure that actually appears under
    "Table 2".
    """

    @staticmethod
    def extract_references(actual_output: str) -> str:
        return f"""You are given a candidate answer. Identify every mention of a named,
document-native structural element (a table, section, footnote, figure, appendix,
or clause referred to by its own name or number, e.g. "Table 3", "Section 4.2",
"footnote 4", "Appendix B") together with the specific claim attached to that
mention.

Do NOT include generic references that don't name a specific labeled element
(e.g. "the document says" or "as shown above" do not count).

Return a JSON object with one key, "references", a list of objects each with:
- "label": the exact structural label mentioned (e.g. "Table 3")
- "claim": the specific claim attached to that label

If the answer makes no reference to any named structural element, return an
empty list.

**
IMPORTANT: Please make sure to only return in JSON format, with the
"references" key as a list of JSON objects.

Example candidate answer: "According to Table 3, EMEA revenue was $12M, and
Section 4.2 sets a 30 day notice period."
Example JSON:
{{
    "references": [
        {{"label": "Table 3", "claim": "EMEA revenue was $12M"}},
        {{"label": "Section 4.2", "claim": "sets a 30 day notice period"}}
    ]
}}
**

Candidate answer:
{actual_output}

JSON:
"""

    @staticmethod
    def generate_verdicts(
        references: List[NamedReference], retrieval_context: List[str]
    ) -> str:
        joined_context = "\n\n".join(retrieval_context)
        references_json = "\n".join(
            f'- label: "{reference.label}", claim: "{reference.claim}"'
            for reference in references
        )

        return f"""You are an attribution judge. You are given a source document (as
retrieved passages, which may preserve the document's own headings, table
captions, section numbers, or footnote markers) and a list of references, each
naming a structural label and a claim attributed to it.

For EACH reference, decide whether the content that actually appears under
that exact label in the source document supports the claim attributed to it.

The 'verdict' key should STRICTLY be either 'yes', 'no', or 'idk':
- 'yes': the label exists in the source and its content supports the claim.
- 'no': the label either does not appear in the source, or its content
  contradicts or does not support the claim (including when the claim is true
  under a *different* label than the one cited).
- 'idk': the source does not contain enough information to decide.

Provide a 'reason' ONLY if the verdict is 'no' or 'idk'. If the claim is true
under a different label than the one cited, name the correct label in the
reason.

**
IMPORTANT: Please make sure to only return in JSON format, with the
'verdicts' key as a list of JSON objects, each with 'label', 'verdict', and
optionally 'reason'.
**

Source document:
{joined_context}

References:
{references_json}

JSON:
"""

    @staticmethod
    def generate_reason(misattributions: List[str], score: str) -> str:
        return f"""Below is a list of named-reference misattributions found in a
candidate answer, and the overall attribution score. Summarize, in one or two
sentences, why the answer received this score. If the list is empty, state
that every named reference was correctly attributed.

Score:
{score}

Misattributions:
{misattributions}

**
IMPORTANT: Please make sure to only return in JSON format, with the 'reason'
key providing the reason.
**

JSON:
"""
