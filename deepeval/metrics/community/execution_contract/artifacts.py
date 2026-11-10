"""Bounded JSON artifacts with atomic writes and corruption detection.

Evidence artifacts deliberately include captured inputs/outputs/metadata and
may contain sensitive values. Reports contain only rule and evidence references.
SHA-256 detects accidental changes; it does not authenticate the producer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import tempfile
from typing import Optional, Union

from .evidence import EvidenceLimits, Snapshot, canonical_json
from .reports import ContractReport
from .rules import Contract, EvaluationLimits
from .serialization import (
    _digest,
    _object,
    contract_from_data,
    contract_to_data,
    report_from_data,
    snapshot_from_data,
    snapshot_to_data,
)

Artifact = Union[Snapshot, Contract, ContractReport]
FORMAT = "deepeval.execution-contract"
VERSION = 1
KINDS = ("snapshot", "contract", "report")


@dataclass(frozen=True)
class ArtifactLimits:
    max_bytes: int = 33554432
    max_json_nodes: int = 1000000
    max_depth: int = 160

    def __post_init__(self):
        for value in vars(self).values():
            if type(value) is not int or value < 1:
                raise ValueError("artifact limits must be positive integers")

    def json_limits(self):
        return EvidenceLimits(
            max_payload_bytes=self.max_bytes,
            max_json_nodes=self.max_json_nodes,
            max_depth=self.max_depth,
        )


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError("nonfinite JSON number")


def encode_artifact(
    value: Artifact,
    *,
    limits: Optional[ArtifactLimits] = None,
    evidence_limits: Optional[EvidenceLimits] = None,
    evaluation_limits: Optional[EvaluationLimits] = None,
) -> bytes:
    limits = limits or ArtifactLimits()
    if type(value) is Snapshot:
        kind, data = "snapshot", snapshot_to_data(value, limits=evidence_limits)
    elif type(value) is Contract:
        kind, data = "contract", contract_to_data(value)
    elif type(value) is ContractReport:
        kind, data = "report", value.to_dict()
        report_from_data(data, limits=evaluation_limits)
    else:
        raise ValueError("unsupported artifact type")
    envelope = {
        "format": FORMAT,
        "version": VERSION,
        "kind": kind,
        "data": data,
    }
    content = canonical_json(envelope, limits.json_limits()).encode("utf-8")
    envelope["sha256"] = hashlib.sha256(content).hexdigest()
    result = (canonical_json(envelope, limits.json_limits()) + "\n").encode(
        "utf-8"
    )
    if len(result) > limits.max_bytes:
        raise ValueError("artifact byte limit exceeded")
    return result


def decode_artifact(
    content: bytes,
    *,
    expected_kind: Optional[str] = None,
    limits: Optional[ArtifactLimits] = None,
    evidence_limits: Optional[EvidenceLimits] = None,
    evaluation_limits: Optional[EvaluationLimits] = None,
) -> Artifact:
    limits = limits or ArtifactLimits()
    if type(content) is not bytes:
        raise ValueError("artifact content must be bytes")
    if len(content) > limits.max_bytes:
        raise ValueError("artifact byte limit exceeded")
    if expected_kind is not None and expected_kind not in KINDS:
        raise ValueError("unknown expected artifact kind")
    try:
        raw = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("invalid artifact JSON") from exc
    _object(raw, ("format", "version", "kind", "data", "sha256"))
    if raw["format"] != FORMAT:
        raise ValueError("unsupported artifact format")
    if type(raw["version"]) is not int or raw["version"] != VERSION:
        raise ValueError("unsupported artifact version")
    kind = raw["kind"]
    if kind not in KINDS or expected_kind is not None and kind != expected_kind:
        raise ValueError("unexpected artifact kind")
    checksum = _digest(raw.pop("sha256"))
    encoded = canonical_json(raw, limits.json_limits()).encode("utf-8")
    if not hmac.compare_digest(hashlib.sha256(encoded).hexdigest(), checksum):
        raise ValueError("artifact checksum mismatch")
    if kind == "snapshot":
        return snapshot_from_data(raw["data"], limits=evidence_limits)
    if kind == "contract":
        return contract_from_data(raw["data"], limits=evaluation_limits)
    return report_from_data(raw["data"], limits=evaluation_limits)


def read_artifact(
    path: Union[str, Path],
    *,
    expected_kind: Optional[str] = None,
    limits: Optional[ArtifactLimits] = None,
    evidence_limits: Optional[EvidenceLimits] = None,
    evaluation_limits: Optional[EvaluationLimits] = None,
) -> Artifact:
    """Read at most max_bytes + 1, including when file metadata is misleading."""
    limits = limits or ArtifactLimits()
    with open(path, "rb") as stream:
        content = stream.read(limits.max_bytes + 1)
    return decode_artifact(
        content,
        expected_kind=expected_kind,
        limits=limits,
        evidence_limits=evidence_limits,
        evaluation_limits=evaluation_limits,
    )


def write_artifact(
    path: Union[str, Path],
    value: Artifact,
    *,
    limits: Optional[ArtifactLimits] = None,
    evidence_limits: Optional[EvidenceLimits] = None,
    evaluation_limits: Optional[EvaluationLimits] = None,
) -> None:
    """Replace a file atomically after fully validating and encoding its value.

    The destination directory must already exist. The temporary file is private
    to its owner, created on the same filesystem, and removed on failure.
    """
    content = encode_artifact(
        value,
        limits=limits,
        evidence_limits=evidence_limits,
        evaluation_limits=evaluation_limits,
    )
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=".execution-contract-",
            delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
