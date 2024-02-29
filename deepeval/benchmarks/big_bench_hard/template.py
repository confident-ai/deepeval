from deepeval.benchmarks.big_bench_hard.task import BigBenchHardTask

class BigBenchHardTemplate:

    @staticmethod
    def generate_output(input, task: BigBenchHardTask):
        if task == BigBenchHardTask.BOOLEAN_EXPRESSIONS:
            return f"""Given the boolean logic, output either 'True' or 'False' as the prediction. No explanations needed.

Boolean Logic:
{input}

Prediction:
"""
        else:
            # TODO: find or implement prompts to conform output
            pass