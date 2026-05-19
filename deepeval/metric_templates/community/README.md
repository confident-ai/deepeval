# Community metric template translations

Shipped non-English metric prompts live here as one JSON file per language (for example `templates.spanish.json`). Each file mirrors the shape of the English [`templates.json`](../templates.json): metric class name → method name → Jinja template string.

## End users

1. Set `DEEPEVAL_METRIC_TEMPLATE_LANGUAGE` to any slug (for example `hindi`, `vietnamese`) in your environment or `.env`.
2. If no templates exist for that language yet, evaluations still run in English and print a **one-time warning** to the console.
3. To add or extend a language locally, run (requires a configured evaluation model):

   ```bash
   deepeval translate vietnamese --metrics "FaithfulnessMetric"
   ```

   This writes **`.deepeval/templates.<lang>.json`**. Re-use the same command to fill in new English methods without overwriting existing keys.

## Contributors

1. Add `LANG = "<lang>"` to `MetricTemplateLanguage` in [`languages.py`](languages.py) (for example `VIETNAMESE = "vietnamese"`).
2. Run translate with **`--contribute`** — `LANG` must match that enum value:

   ```bash
   deepeval translate vietnamese --metrics "FaithfulnessMetric" --contribute
   ```

3. Commit `templates.<lang>.json` and the enum change, then open a pull request.

## Resolution at runtime

English prompts always come from [`templates.json`](../templates.json). For a non-English slug, deepeval loads shipped `community/templates.<lang>.json` and merges **`.deepeval/templates.<lang>.json`** on top (local keys win). Missing class/method entries fall back to English with a one-time console warning per metric class.
