"""The legacy-signature retry path must still advance the progress bar."""

import asyncio

import pytest
from rich.progress import Progress

import deepeval.metrics.indicator as indicator


class _LegacyMetric:
	"""Pre-v2 signature: a_measure takes no extra kwargs, so the first
	keyword-bearing call raises TypeError and safe_a_measure retries bare."""

	measured = False

	async def a_measure(self, test_case) -> None:
		_LegacyMetric.measured = True


async def test_retry_path_advances_progress_bar(monkeypatch):
	calls: list = []
	monkeypatch.setattr(indicator, "update_pbar", lambda *a, **k: calls.append(a))

	metric = _LegacyMetric()
	await indicator.safe_a_measure(
		metric,
		object(),
		ignore_errors=True,
		skip_on_missing_params=False,
		progress=Progress(),
		pbar_eval_id=7,
	)

	assert _LegacyMetric.measured  # retry actually ran
	assert calls == [(Progress(), 7)] or len(calls) == 1 and calls[0][1] == 7
