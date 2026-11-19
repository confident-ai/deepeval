"""Where an agent turn's text came from.

The connector may carry the agent's own transcript of what it said (LiveKit and
Pipecat agents publish one); otherwise, or when a barge cut the reply short,
the simulator transcribes the agent's audio itself. Recording which one wrote
`content` lets a reader judge how far to trust it.
"""

from typing import Any, Dict, Optional

TRANSCRIPT_SOURCE_KEY = "transcript_source"
TRANSCRIPT_FROM_AGENT = "agent"
TRANSCRIPT_FROM_STT = "transcription"


def transcript_source_metadata(
    source: Optional[str], metadata: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    if source is None:
        return metadata
    return {**(metadata or {}), TRANSCRIPT_SOURCE_KEY: source}
