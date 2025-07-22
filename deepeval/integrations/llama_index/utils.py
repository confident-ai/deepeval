from deepeval.test_case.llm_test_case import LLMTestCase
from deepeval.tracing.types import BaseSpan

def parse_id(id_: str) -> tuple[str, str]:
    """
    Parse the id_ into a tuple of class name and method name, ignoring any suffix after '-'.
    """
    # Ignore everything after the first '-'
    main_part = id_.split("-", 1)[0]
    # Split by '.' to get class and method
    class_name, method_name = main_part.rsplit(".", 1)
    return class_name, method_name

def prepare_input_llm_test_case_params(class_name: str, method_name: str, span: BaseSpan, args: dict):
    
    # condition for parent agent span
    if class_name == "Workflow" and method_name == "run":
        start_event = args.get("start_event")
        
        from llama_index.core.agent.workflow.workflow_events import AgentWorkflowStartEvent
        if isinstance(start_event, AgentWorkflowStartEvent):
            input = ""
            for key, value in start_event.items():
                input += f"{key}: {value}\n"
            
            span.llm_test_case = LLMTestCase(
                input=input,
                actual_output="",
            )
