"""Deterministic reason strings for ``TrajectoryDivergenceMetric``.

Unlike most metric templates, this module renders no LLM prompt: the
trajectory divergence metric is fully local and deterministic. These are
sentence builders that turn an ``AlignmentResult`` into a human-readable
explanation of *which* step diverged and *how*, keeping the difference kind
and the recovery status separate so a recovered retry is never mistaken for
an unrecovered fork.
"""


class TrajectoryDivergenceTemplate:
    """Sentence builders for the trajectory divergence reason text."""

    @staticmethod
    def aligned(num_steps: int) -> str:
        noun = "step" if num_steps == 1 else "steps"
        return (
            "The baseline and candidate trajectories are aligned across all "
            f"{num_steps} {noun}."
        )

    @staticmethod
    def arg_change(step: int, tool: str) -> str:
        return (
            f"step {step} calls the same tool `{tool}` with different "
            "arguments"
        )

    @staticmethod
    def tool_change(step: int, baseline_tool: str, candidate_tool: str) -> str:
        return (
            f"step {step} calls `{baseline_tool}` in the baseline but "
            f"`{candidate_tool}` in the candidate"
        )

    @staticmethod
    def order_change(first_step: int, last_step: int) -> str:
        return (
            f"steps {first_step}-{last_step} contain the same calls but in "
            "a different order"
        )

    @staticmethod
    def absent(step: int, tool: str) -> str:
        return f"step {step} (`{tool}`) is absent from the candidate trace"

    @staticmethod
    def extra(step: int, tool: str) -> str:
        return (
            f"the candidate trace inserts an extra step at step {step} "
            f"(`{tool}`)"
        )

    @staticmethod
    def recovered(resync_step: int) -> str:
        return (
            f"The trajectories resynchronize at step {resync_step}: the "
            "divergence is localized to the intervening steps rather than a "
            "fork to the end."
        )

    @staticmethod
    def unrecovered() -> str:
        return (
            "The trajectories do not resynchronize: the divergence persists "
            "to the end of the trace."
        )
