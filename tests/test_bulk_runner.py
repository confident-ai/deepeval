# from deepeval.bulk_runner import BulkTestRunner, TestCase


# def generate_llm_output(input: str):
#     expected_output = "Our customer success phone line is 1200-231-231."
#     return expected_output


# def test_bulk_runner():
#     class BulkTester(BulkTestRunner):
#         @property
#         def bulk_test_cases(self):
#             return [
#                 TestCase(
#                     input="What is the customer success number",
#                     expected_output="1800-213-123",
#                 )
#             ]

#     tester = BulkTester()
#     tester.run(completion_fn=generate_llm_output)
