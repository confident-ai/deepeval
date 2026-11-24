from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

# What an event's place on the call covers.
MOMENT = "moment"
RANGE = "range"
WHOLE_REPLY = "whole_reply"


class VoiceEvent(BaseModel):
    """One problem a voice metric found on a call.

    Every voice metric reports its events in this shape. Where the problem
    is known to a point in time, `offset_seconds` and `duration_seconds`
    place it inside the turn's audio and `start_time` places it on the
    call's clock. `mark` says what that place covers: the spot where the
    problem happened (`moment`), the stretch it lasted (`range`), or the
    whole agent reply it was judged on (`whole_reply`). `timing_source` says
    whether the place was read from the stereo call recording or from turn
    timestamps, and `description` is one sentence about the problem. What a
    metric measured about the problem (a count, a level, a delay) is kept
    alongside these fields under its own name.
    """

    model_config = ConfigDict(extra="allow")

    type: str
    turn: Optional[int] = None
    role: Optional[str] = None
    offset_seconds: Optional[float] = None
    duration_seconds: Optional[float] = None
    start_time: Optional[float] = None
    mark: Optional[str] = None
    timing_source: Optional[str] = None
    description: Optional[str] = None

    def get(self, field: str, default: Any = None) -> Any:
        return getattr(self, field, default)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


def placed(audio: Any, offset: float, duration: float) -> Dict[str, Any]:
    """Where a problem sits inside a turn's audio, and on the call's clock
    when the clip has a start time."""
    start = getattr(audio, "start_time", None)
    return {
        "offset_seconds": round(offset, 3),
        "duration_seconds": round(duration, 3),
        "start_time": None if start is None else round(start + offset, 3),
    }


def on_call(start: float, duration: float) -> Dict[str, Any]:
    """Where a problem sits on the call's clock."""
    return {
        "start_time": round(start, 3),
        "duration_seconds": round(duration, 3),
    }


def events_to_dicts(events: List[VoiceEvent]) -> List[Dict[str, Any]]:
    return [event.to_dict() for event in events]
