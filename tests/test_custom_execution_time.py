# from deepeval.metrics import BaseMetric
# from deepeval.test_case import LLMTestCase
# from deepeval import assert_test

# class ExecutionTimeMetric(BaseMetric):
#     def __init__(self, max_execution_time: float):
#         self.max_execution_time = max_execution_time
    
#     def measure(self, test_case: LLMTestCase):
#         self.success = test_case.execution_time <= self.max_execution_time
#         if self.success:
#             self.score = 1
#         else:
#             self.score = 0

#         return self.score
    
#     def is_successful(self):
#         return self.success

#     @property
#     def name(self):
#         return "Execution Time"
    

# def test_execution_time():
#     test_case = LLMTestCase(input="...", actual_output="...", execution_time=4.57)
#     execution_time_metric = ExecutionTimeMetric(max_execution_time=5)
#     assert_test(test_case, [execution_time_metric])