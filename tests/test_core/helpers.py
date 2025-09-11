def teardown_settings_singleton():
    import deepeval.config.settings as settings_mod

    settings_mod._settings_singleton = None


def reset_settings_env(monkeypatch, *, skip_keys: set[str] = set()):
    # reset singleton
    teardown_settings_singleton()

    # drop env vars that map to Settings fields
    from deepeval.config.settings import Settings

    for k in Settings.model_fields.keys():
        if k not in skip_keys:
            monkeypatch.delenv(k, raising=False)

    # don’t carry default save across tests, keep things clean
    monkeypatch.delenv("DEEPEVAL_DEFAULT_SAVE", raising=False)
