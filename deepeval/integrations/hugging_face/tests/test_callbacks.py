"""Test for callbacks"""

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
)


from deepeval.integrations.hugging_face import DeepEvalHuggingFaceCallback
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset, Golden

import os
import random

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["OPENAI_API_KEY"] = "API-KEY"


def create_prompt(row):
    """Merge Context and Question into a single string"""
    contexts = row["context"]["contexts"]
    question = row["question"]
    prompt = f"""{'CONTEXT: ' + str("; ".join(contexts)) if contexts else ''}
            QUESTION: {question}
            ANSWER:"""
    return {"input": prompt, "response": row["long_answer"]}


def prepare_dataset(tokenizer, tokenizer_args):
    from datasets import load_dataset

    dataset = load_dataset("pubmed_qa", "pqa_labeled")
    merged_dataset = dataset.map(
        create_prompt,
        remove_columns=[
            "question",
            "context",
            "long_answer",
            "pubid",
            "final_decision",
        ],
    )

    def tokenize_text(dataset, padding="max_length"):
        model_input = tokenizer(dataset["input"], **tokenizer_args)
        response = tokenizer(dataset["response"], **tokenizer_args)

        if padding == "max_length":
            response["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in response["input_ids"]
            ]

        model_input["labels"] = response["input_ids"]
        return model_input

    tokenized_dataset = merged_dataset.map(
        tokenize_text, remove_columns=["input", "response"]
    )
    tokenized_dataset = tokenized_dataset.map(
        lambda x: {
            "input_ids": x["input_ids"][0],
            "labels": x["labels"][0],
            "attention_mask": x["attention_mask"][0],
        }
    )
    return dataset, merged_dataset, tokenized_dataset


def create_deepeval_dataset(dataset, sample_size):
    total_length = len(dataset)
    random_index_list = [
        random.randint(0, total_length) for _ in range(sample_size)
    ]
    eval_dataset = [dataset[row] for row in random_index_list]
    goldens = []
    for row in eval_dataset:
        context = ["; ".join(row["context"]["contexts"])]
        golden = Golden(
            input=row["question"],
            expected_output=row["long_answer"],
            context=context,
            retrieval_context=context,
        )
        goldens.append(golden)

    return EvaluationDataset(goldens=goldens)


if __name__ == "__main__":
    # initialize tokenizer
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

    # initalize model
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    model.resize_token_embeddings(len(tokenizer))

    # create tokenized dataset
    tokenizer_args = {
        "return_tensors": "pt",
        "max_length": 128,
        "padding": "max_length",
        "truncation": True,
        "padding": True,
    }

    dataset, merged_dataset, tokenized_dataset = prepare_dataset(
        tokenizer, tokenizer_args
    )

    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
    )

    repository_id = f"flan-t5-small"

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=repository_id,
        overwrite_output_dir=True,
        num_train_epochs=50,
        per_device_train_batch_size=8,
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
    )

    eval_dataset = create_deepeval_dataset(dataset["train"], sample_size=5)
    hallucination_metric = HallucinationMetric(threshold=0.3)
    answer_relevancy_metric = AnswerRelevancyMetric(
        threshold=0.5, model="gpt-3.5-turbo"
    )
    metrics = [hallucination_metric, answer_relevancy_metric]

    # initalize DeepEvalHuggingFaceCallback
    callback = DeepEvalHuggingFaceCallback(
        metrics=metrics,
        evaluation_dataset=eval_dataset,
        tokenizer_args=tokenizer_args,
        trainer=trainer,
        show_table=True,
    )
    trainer.add_callback(callback)
    trainer.train()
