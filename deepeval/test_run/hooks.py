on_test_run_end_callback = None


def on_test_run_end(func):
    global on_test_run_end_callback
    on_test_run_end_callback = func

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def invoke_test_run_end_callback():
    global on_test_run_end_callback
    if on_test_run_end_callback:
        on_test_run_end_callback()
        on_test_run_end_callback = None
