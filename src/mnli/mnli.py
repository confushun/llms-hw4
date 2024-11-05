from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any


def load_mnli_train() -> Dataset:
    """you can use this for coming up with a few-shot prompt."""
    
    ds = load_dataset("facebook/anli", split="train_r3").take(100)
    return ds

def load_mnli_dev() -> Dataset:
    """Use this for picking your best verbalizer."""
    
    ds = load_dataset("facebook/anli", split="dev_r3").shuffle().take(50)
    return ds

def load_mnli_test() -> Dataset:
    """Use this only AFTER you have picked your best verbalizer."""
    
    ds = load_dataset("facebook/anli", split="test_r3").shuffle().take(100)
    return ds

def make_verbalizer(dev_ds:Dataset) -> str:
    """Should return a verbalizer string. You may choose to use examples from the dev set in the verbalizer."""

    # YOUR CODE HERE
    pass

def make_prompt(verbalizer: str, premise: str, hypothesis:str) -> str:
    """Given a verbalizer, a premise, and a hypothesis, return the prompt."""
    
    # YOUR CODE HERE
    pass

def predict_labels(prompts: list[str]):
    """Should return a list of integer predictions (0, 1 or 2), one per prompt."""
    
    # YOUR CODE HERE.
    pass

if __name__ == "__main__":
    train_ds = load_mnli_train()
    dev_ds = load_mnli_dev()
    test_ds = load_mnli_test()
    
    verbalizer = make_verbalizer(train_ds)
    
    prompts = []
    true_labels = []
    for ex in dev_ds:
      prompt = make_prompt(verbalizer, ex["premise"], ex["hypothesis"])
      prompts.append(prompt)
      true_labels.append(ex["label"])
    
    predicted_labels = predict_labels(prompts)
    
    num_correct = sum(true == pred for true, pred in zip(true_labels, predicted_labels))
    accuracy = num_correct / len(true_labels)
    print("Accuracy:", accuracy)