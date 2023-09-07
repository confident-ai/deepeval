# QuickStart

Once you have installed, run the login command. During this step, you will be asked to visit https://app.confident-ai.com to grab your API key.

Note: this step is entirely optional if you do not wish to track your results but we highly recommend it so you can view how results differ over time.

```bash
deepeval login

# If you already have an API key
deepeval login --api-key $API_KEY
```

Once you have logged in, you can generate a sample test file as shown below. This test file allows you to quickly get started modifying it with various tests. (More on this later)

```bash
deepeval test generate --output-file test_sample.py
```

Once you have generated the test file, you can then run tests as shown.

```bash
deepeval test run test_sample.py
```

## About the sample test file 

The sample test file that you have generated is highly important.

