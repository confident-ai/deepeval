"""Test for callbacks
"""

from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling

import datasets
import json

from deepeval.callbacks.huggingface import DeepEvalCallback
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

# load dataset
f = open(r"D:\deepeval-callback\deepeval\build\ra_top_1000_data_set.json", 'r', encoding='utf-8').read()
data = json.loads(f)
final_data = {'text': [x['bio'] for x in data][:200]}
dataset = datasets.Dataset.from_dict(final_data)

# initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/gpt-neo-125M",
    bos_token='<|startoftext|>', 
    eos_token='<|endoftext|>', 
    pad_token='<|pad|>'
)

# initalize model
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
model.resize_token_embeddings(len(tokenizer))

# create tokenized dataset
tokenizer_args = {
    "return_tensors":"pt", 
    "max_length": 64, 
    "padding": "max_length", 
    "truncation": True
}

def tokenize_function(examples):
    return tokenizer(examples["text"], **tokenizer_args)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# create LLMTestCases
first_test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra costs.", 
    context=["All customers are eligible for a 30 day full refund at no extra costs."]
)
second_test_case = LLMTestCase(
    input="What if these shoes don't fit?", 
    actual_output="We also sell 20 gallons of pepsi", 
    context=["All customers are eligible for a 30 day full refund at no extra costs."]
)

# create deepeval metrics list
dataset = EvaluationDataset(test_cases=[first_test_case, second_test_case])
hallucination_metric = HallucinationMetric(minimum_score=0.3)
answer_relevancy_metric = AnswerRelevancyMetric(minimum_score=0.5)
metrics = [hallucination_metric, answer_relevancy_metric]

# initalize training_args
training_args = TrainingArguments(
    output_dir="./gpt2-fine-tuned",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=8
)

# initalize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets
)

# initalize DeepEvalCallback
callback = DeepEvalCallback(
    metrics=metrics, 
    evaluation_dataset=dataset, 
    tokenizer_args=tokenizer_args,
    trainer=trainer,
    show_table=True,
    show_table_every=1
)
trainer.add_callback(callback)
trainer.train()