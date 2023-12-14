on_test_run_end_hook = None


def on_test_run_end(func):
    global on_test_run_end_hook
    on_test_run_end_hook = func

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def invoke_test_run_end_hook():
    global on_test_run_end_hook
    if on_test_run_end_hook:
        on_test_run_end_hook()
        on_test_run_end_hook = None
