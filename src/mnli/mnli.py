
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any

from olmo.inference import load_model_and_generate_output,hit_vllm_model_and_generate_output


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

    return ('Given below is a [premise] and [hypothesis]. Classify the label of the [hypothesis] with respect to the [premise]. '
            'If the [hypothesis] contradicts the [premise], output the label as 2. If the [hypothesis] is implied by the [premise],'
            ' output the label as 0. If it neither contradicts nor is implied, output the label as 1. ')

def make_prompt(verbalizer: str, premise: str, hypothesis:str) -> str:
    """Given a verbalizer, a premise, and a hypothesis, return the prompt."""
    return f'{verbalizer}. \n \n premise: {premise} \n\n hypothesis: {hypothesis} \n\n What is the label? Only give me the number.'

def predict_labels(prompts: list[str]):
    """Should return a list of integer predictions (0, 1 or 2), one per prompt."""
    
    results = []
    invalid_labels = 0
    for prompt in prompts:
        try:
            label = prompt.replace(" ", "").replace(".", "")
            results.append(int(label))
        except:
            invalid_labels += 1
            print(f'{prompt} is not a valid label')
            results.append(-1)

    print(f'invalid labels: {invalid_labels}')

    return results

# input message -> change to user, content json format --> apply chat template ---> send it to the model
# ---> model output
# model


# here:
# prompt
# add verbalizer + prompt + label?
# send this to the model as input msg
# follow same as above
# change output to the numerical label

if __name__ == "__main__":
    train_ds = load_mnli_train()
    dev_ds = load_mnli_dev()
    test_ds = load_mnli_test()
    verbalizer = make_verbalizer(train_ds)

    prompts = []
    true_labels = []

    # make input prompts
    for ex in dev_ds:
      prompt = make_prompt(verbalizer, ex["premise"], ex["hypothesis"])
      prompts.append(prompt)
      true_labels.append(ex["label"])

    #print(f'prompts: {prompts}')

    # generate outputs
    outputs= hit_vllm_model_and_generate_output(prompts)
    print(f'outputs: {outputs}')

    # fetch their labels
    predicted_labels = predict_labels(outputs)
    print(f'predicted labels: {predicted_labels}')
    print(f'true labels: {true_labels}')

    num_correct = sum(true == pred for true, pred in zip(true_labels, predicted_labels))
    accuracy = num_correct / len(true_labels)
    print("Accuracy:", accuracy)