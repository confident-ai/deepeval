from typing import Dict, List

ENTERPRISE_TEXT = """
This document is intended to provide general guidance on standard operational practices, internal processes, and shared responsibilities across teams. The content is designed to support consistency and clarity when handling routine tasks, reviewing inputs, and generating outputs based on provided information.

Processes described here may apply differently depending on context, role, or environment. Teams are encouraged to use judgment when interpreting guidance and to consider applicable constraints, dependencies, or downstream impacts. Where uncertainty exists, escalation through appropriate channels is recommended.

The examples and descriptions included are illustrative only and are not intended to represent real individuals, organizations, or systems. Information may be updated periodically to reflect changes in tooling, workflows, or organizational priorities. Readers should ensure they are working from the most recent version available.

In general, adherence to documented processes helps reduce ambiguity, improve reliability, and support alignment across stakeholders. Regular review of internal materials is encouraged to maintain accuracy and relevance over time.
"""
LOG_TEXT = """
[2024-03-18 10:32:11] INFO  Request received and queued for processing
[2024-03-18 10:32:11] INFO  Input validation completed without errors
[2024-03-18 10:32:12] INFO  Default configuration parameters applied
[2024-03-18 10:32:12] INFO  Processing step completed successfully
[2024-03-18 10:32:12] INFO  Intermediate output generated
[2024-03-18 10:32:13] INFO  No warnings or exceptions detected
[2024-03-18 10:32:13] INFO  Response prepared for downstream handling
[2024-03-18 10:32:13] INFO  Execution finished with status=success
"""
TRANSCRIPT_TEXT = """
Speaker 1: Thanks everyone for joining today. The goal of this discussion is to review recent updates, align on current priorities, and identify any open questions.

Speaker 2: That makes sense. I went through the materials earlier and noted a few areas where clarification might be helpful, especially around how certain cases are handled.

Speaker 1: We can definitely walk through those. In general, the approach has been to follow existing guidelines while allowing flexibility when context requires it.

Speaker 3: From what I've seen, the guidance has been useful, though there are still some edge cases that come up depending on the situation.

Speaker 2: Agreed. It would be helpful to capture those examples and refine the documentation over time.
"""


DEFAULT_PREFIXES = {
    "enterprise": ENTERPRISE_TEXT,
    "logs": LOG_TEXT,
    "transcript": TRANSCRIPT_TEXT,
}
