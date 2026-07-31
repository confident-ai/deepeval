"""Anonymous identity, stored once per machine rather than once per folder.

The store lives in the user's home directory. It used to sit in a CWD-relative
`.deepeval/`, which minted a fresh id for every project folder and every
container. Home only fixes the folder case -- ephemeral infrastructure still
churns -- but it is the half that is fixable.
"""

import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from deepeval.constants import HIDDEN_DIR
from deepeval.telemetry.events import Feature
from deepeval.telemetry.properties import TelemetryKey, UserStatus

TELEMETRY_DATA_FILE = ".deepeval_telemetry.txt"

OPTED_OUT_ID = "telemetry-opted-out"

_lock = threading.Lock()
_data: Optional[Dict[str, str]] = None
_identity: Optional["Identity"] = None


@dataclass(frozen=True)
class Identity:
    anonymous_id: str
    status: UserStatus


def _home_dir() -> Path:
    override = os.getenv("DEEPEVAL_HOME")
    if override:
        return Path(override)
    return Path.home() / ".deepeval"


def telemetry_path() -> Path:
    return _home_dir() / TELEMETRY_DATA_FILE


def _legacy_paths() -> List[Path]:
    """Where the id lived before it moved to the home directory."""
    return [Path(HIDDEN_DIR) / TELEMETRY_DATA_FILE, Path(TELEMETRY_DATA_FILE)]


def _parse(path: Path) -> Dict[str, str]:
    try:
        with open(path, "r") as handle:
            lines = handle.readlines()
    except OSError:
        return {}
    data: Dict[str, str] = {}
    for line in lines:
        key, _, value = line.strip().partition("=")
        if key:
            data[key] = value
    return data


def _legacy_feature_key(feature: Feature) -> str:
    """The pre-v2 `DEEPEVAL_{FEATURE}_STATUS` key this feature used to use."""
    return f"DEEPEVAL_{feature.value.upper()}_STATUS"


def _adopt_legacy_feature_keys(data: Dict[str, str]) -> bool:
    """Fold the pre-v2 per-feature keys into the single seen-features list.

    Without this every existing user reports `feature.status = new` once per
    feature on their first v2 run, which would read as a wave of new adoption
    at rollout rather than the same people doing the same things.
    """
    seen = _seen_features(data)
    changed = False
    for feature in Feature:
        # Presence is the signal, whatever the recorded value: the key only
        # ever existed because the feature had been used at least once.
        if data.pop(_legacy_feature_key(feature), None) is None:
            continue
        changed = True
        seen.add(feature.value)
    if changed:
        data[TelemetryKey.SEEN_FEATURES.value] = ",".join(sorted(seen))
    return changed


def _load() -> Dict[str, str]:
    """Read the store, migrating a legacy CWD file the first time we see one."""
    path = telemetry_path()
    if path.exists():
        data = _parse(path)
        if _adopt_legacy_feature_keys(data):
            _persist(data)
        return data
    for legacy in _legacy_paths():
        if legacy.exists():
            data = _parse(legacy)
            if data.get(TelemetryKey.ID.value):
                _adopt_legacy_feature_keys(data)
                _persist(data)
                return data
    return {}


def _persist(data: Dict[str, str]) -> None:
    from deepeval.telemetry.client import telemetry_opt_out

    if telemetry_opt_out():
        return
    try:
        path = telemetry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            for key, value in data.items():
                handle.write(f"{key}={value}\n")
    except OSError:
        # A read-only or unwritable home must never break an evaluation.
        pass


def read_telemetry_file() -> Dict[str, str]:
    global _data
    with _lock:
        if _data is None:
            _data = _load()
        return dict(_data)


def write_telemetry_file(data: Dict[str, str]) -> None:
    global _data
    with _lock:
        if _data == data:
            return
        _data = dict(data)
        _persist(_data)


def get_identity() -> Identity:
    """Resolve the id and first-run status together, exactly once.

    Reading status separately from the id is what made the first *two* events
    of a fresh install both report `new`: the old `get_status()` ran before
    `get_unique_id()` had written the flag.
    """
    global _identity
    from deepeval.telemetry.client import telemetry_opt_out

    if telemetry_opt_out():
        return Identity(anonymous_id=OPTED_OUT_ID, status=UserStatus.OLD)

    with _lock:
        if _identity is not None:
            return _identity

    data = read_telemetry_file()
    anonymous_id = data.get(TelemetryKey.ID.value)
    if anonymous_id:
        status = UserStatus.OLD
    else:
        anonymous_id = str(uuid.uuid4())
        status = UserStatus.NEW
        data[TelemetryKey.ID.value] = anonymous_id
        # Persist "old" now so the next process reports old, not this one.
        data[TelemetryKey.STATUS.value] = UserStatus.OLD.value
        write_telemetry_file(data)

    identity = Identity(anonymous_id=anonymous_id, status=status)
    with _lock:
        _identity = identity
    return identity


def get_unique_id() -> str:
    return get_identity().anonymous_id


def get_status() -> str:
    return get_identity().status.value


def _seen_features(data: Dict[str, str]) -> Set[str]:
    raw = data.get(TelemetryKey.SEEN_FEATURES.value, "")
    return {item for item in raw.split(",") if item}


def get_feature_status(feature: Feature) -> UserStatus:
    data = read_telemetry_file()
    return (
        UserStatus.OLD
        if feature.value in _seen_features(data)
        else UserStatus.NEW
    )


def set_last_feature(feature: Feature) -> None:
    data = read_telemetry_file()
    data[TelemetryKey.LAST_FEATURE.value] = feature.value
    seen = _seen_features(data)
    seen.add(feature.value)
    data[TelemetryKey.SEEN_FEATURES.value] = ",".join(sorted(seen))
    write_telemetry_file(data)


def get_last_feature() -> Feature:
    data = read_telemetry_file()
    last_feature = data.get(TelemetryKey.LAST_FEATURE.value)
    try:
        return Feature(last_feature)
    except ValueError:
        return Feature.UNKNOWN


def set_logged_in_with(logged_in_with: str) -> None:
    """Record the account locally. The address itself is never transmitted.

    Sending it, and calling `identify`, would attach a real email to every
    event -- which the published privacy page does not disclose. Only the
    boolean in `is_logged_in` leaves the machine.
    """
    data = read_telemetry_file()
    data[TelemetryKey.LOGGED_IN_WITH.value] = logged_in_with
    write_telemetry_file(data)


def get_logged_in_with() -> str:
    data = read_telemetry_file()
    return data.get(TelemetryKey.LOGGED_IN_WITH.value, "NA")


def is_logged_in() -> bool:
    return get_logged_in_with() != "NA"


def reset_cache_for_testing() -> None:
    global _data, _identity
    with _lock:
        _data = None
        _identity = None
