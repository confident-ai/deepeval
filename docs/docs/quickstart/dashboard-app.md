# View Your Dashboard

DeepEval provides an easy way log metrics to the dashboard.

You can see a preview of our dashboard below:

![Dashboard Example](../../assets/dashboard-screenshot.png)

Our dashboard allows you to quickly compare the output and expected output of results.

Access the dashboard by simply setting your API key as an environment variable as below.

## Get started

[Get your API key here](https://app.confident-ai.com)

```python
os.environ["CONFIDENT_AI_API_KEY"] = "xxx"
# Name your implementation - e.g. "LangChain Implementation"
os.environ["CONFIDENT_AI_IMP_NAME"] = "QuickStart"
```

Once you set the API key - we automatically log metrics to our server.

### Turning it off

If at some point you decide you no longer want to log metrics to our server, you can easily turn it off by setting another environment variable (or corrupting the API key).

```python
os.environ["DO_NOT_SEND_TO_CONFIDENT_AI"] = "Y"
```

If you require a custom deployment fo DeepEval inside of your infrastructure, please contact jacky@twilix.io
