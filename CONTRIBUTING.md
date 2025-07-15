# Contributing to DeepEval 🥳

Thanks for thinking about contributing to DeepEval! We accept fixes, improvements, or even entire new features. Some reasons why you might want to contribute:

- there's a bug that you want fixed
- there's a cool new feature you're thinking about that might be useful for DeepEval
- there's a metric or benchmark that you want implemented
- there's room for improvement in the docs

## How to contribute

We follow fork and pull request workflow. To know more about it, check out this [guide](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).

### Set up your development environment

1. Create a python virtual environment.
2. We recommend using poetry to install dependencies. If you haven't already, you can install it [here's the guide](https://python-poetry.org/docs/).
3. Install the dependencies using:

```bash
make install
```

## Our expectations (not a lot :)

To contribute, all we ask for is to follow existing patterns within the codebase. For example, if you're looking to add a new benchmark, look at how the different modules in the existing benchmarks are structured and implemented, and we encourage you to reuse helper functions and methods shared by similar modules.

Other than that, there are no strict rules to follow, except for optionally running `black` to ensure good formatting. Also, there's no need to worry about failing test cases in Github Actions, as these are mostly for internal use and will only pass if triggered by a user with the correct permissions within Confident AI.

Thank you and come ask any questions or discuss any new PRs you have in mind at our [discord](https://discord.com/invite/a3K9c8GRGt)!
