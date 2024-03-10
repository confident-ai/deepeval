from deepeval.benchmarks.big_bench_hard.task import BigBenchHardTask
from deepeval.benchmarks.big_bench_hard.cot_prompts import *
from deepeval.benchmarks.big_bench_hard.shot_prompts import *

class BigBenchHardTemplate:

    @staticmethod
    def generate_output(input: str, task: BigBenchHardTask, n_shots: int, enable_cot: bool):
        
        # generate folder path based on enable_cot
        path = "./"
        if enable_cot:
            path += "cot_prompts/"
        else:
            path += "shot_prompts/"
        
        # get prompt from text file based on n_shots and folder path
        path += BigBenchHardTemplate.get_filename(task)
        prompt = 'Task description: '
        prompt += '\n\n'.join(BigBenchHardTemplate.read_file(path)[: n_shots + 1])
        prompt += '\n\nQ: ' + input + '\nA: ' 
        
        return prompt


    # a function to 
    def read_file(file_path):
        # Open the file in read mode ('r')
        with open(file_path, 'r') as file:
            # Read the entire content of the file
            file_content = file.read()

        # Now you can use file_content as you need
        sections = file_content.split('\n\n')
        return sections
    
    def get_filename(task):
        # generate prompts
        if task == BigBenchHardTask.BOOLEAN_EXPRESSIONS:
            return 'boolean_expressions.txt'
        if task == BigBenchHardTask.CAUSAL_JUDGEMENT:
            return 'causal_judgement.txt'
        elif task == BigBenchHardTask.DATE_UNDERSTANDING:
            return 'date_understanding.txt'
        elif task == BigBenchHardTask.DISAMBIGUATION_QA:
            return 'disambiguation_qa.txt'
        elif task == BigBenchHardTask.DYCK_LANGUAGES:
            return 'dyck_languages.txt'
        elif task == BigBenchHardTask.FORMAL_FALLACIES:
            return 'formal_fallacies.txt'
        elif task == BigBenchHardTask.GEOMETRIC_SHAPES:
            return 'geometric_shapes.txt'
        elif task == BigBenchHardTask.HYPERBATON:
            return 'hyperbaton.txt'
        elif task == BigBenchHardTask.LOGICAL_DEDUCTION_FIVE_OBJECTS:
            return 'logical_deduction_five_objects.txt'
        elif task == BigBenchHardTask.LOGICAL_DEDUCTION_SEVEN_OBJECTS:
            return 'logical_deduction_seven_objects.txt'
        elif task == BigBenchHardTask.LOGICAL_DEDUCTION_THREE_OBJECTS:
            return 'logical_deduction_three_objects.txt'
        elif task == BigBenchHardTask.MOVIE_RECOMMENDATION:
            return 'movie_recommendation.txt'
        elif task == BigBenchHardTask.MULTISTEP_ARITHMETIC_TWO:
            return 'multistep_arithmetic_two.txt'
        elif task == BigBenchHardTask.NAVIGATE:
            return 'navigate.txt'
        elif task == BigBenchHardTask.OBJECT_COUNTING:
            return 'object_counting.txt'
        elif task == BigBenchHardTask.PENGUINS_IN_A_TABLE:
            return 'penguins_in_a_table.txt'
        elif task == BigBenchHardTask.REASONING_ABOUT_COLORED_OBJECTS:
            return 'reasoning_about_colored_objects.txt'
        elif task == BigBenchHardTask.RUIN_NAMES:
            return 'ruin_names.txt'
        elif task == BigBenchHardTask.SALIENT_TRANSLATION_ERROR_DETECTION:
            return 'salient_translation_error_detection.txt'
        elif task == BigBenchHardTask.SNARKS:
            return 'snarks.txt'
        elif task == BigBenchHardTask.SPORTS_UNDERSTANDING:
            return 'sports_understanding.txt'
        elif task == BigBenchHardTask.TEMPORAL_SEQUENCES:
            return 'temporal_sequences.txt'
        elif task == BigBenchHardTask.TRACKING_SHUFFLED_OBJECTS_FIVE_OBJECTS:
            return 'tracking_shuffled_objects_five_objects.txt'
        elif task == BigBenchHardTask.TRACKING_SHUFFLED_OBJECTS_SEVEN_OBJECTS:
            return 'tracking_shuffled_objects_seven_objects.txt'
        elif task == BigBenchHardTask.TRACKING_SHUFFLED_OBJECTS_THREE_OBJECTS:
            return 'tracking_shuffled_objects_three_objects.txt'
        elif task == BigBenchHardTask.WEB_OF_LIES:
            return 'web_of_lies.txt'
        elif task == BigBenchHardTask.WORD_SORTING:
            return 'word_sorting.txt'
        
