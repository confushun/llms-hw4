import numpy as np
from tqdm import tqdm
import litellm
import openai
from datasets import load_dataset
import os
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Union, Any, Optional


load_dotenv()


def read_data(seed: int, train_size: int, test_size: int) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Load and prepare the SST-2 dataset.

    Args:
        seed (int): Random seed for dataset shuffling
        train_size (int): Number of training samples to select
        test_size (int): Number of testing samples to select

    Returns:
        Tuple containing:
            - List[str]: Training sentences
            - List[int]: Training labels (0 for negative, 1 for positive)
            - List[str]: Test sentences
            - List[int]: Test labels (0 for negative, 1 for positive)
    """

    dataset = load_dataset("stanfordnlp/sst2")

    # Get training and testing datasets from `train` and `validation` splits
    # Step 1: shuffle with seed
    # Step 2: select first train/test size data
    train_data = ...
    test_data = ...

    # Extract sentences and labels
    train_sentences = ...
    train_labels = ...
    test_sentences = ...
    test_labels = ...

    return train_sentences, train_labels, test_sentences, test_labels


def create_prompt(
    q_prefix: str,
    a_prefix: str,
    few_shot_sentences: List[str], 
    few_shot_labels: List[str], 
    test_sentence: str
) -> str:
    """
    Create a prompt for sentiment analysis using few-shot examples.

    Args:
        few_shot_sentences (List[str]): List of example sentences for few-shot learning
        few_shot_labels (List[str]): List of corresponding labels (0/1)
        test_sentence (str): The sentence to analyze

    Returns:
        str: Formatted prompt string containing examples and test sentence
    """

    prompt = ""

    # Add few-shot samples
    for s, l in zip(few_shot_sentences, few_shot_labels):
        if isinstance(l, int):
            l = "Positive" if l == 1 else "Negative"
        prompt += f"{q_prefix}{s}\n{a_prefix}{l}\n\n"

    # Add test sentence
    prompt += f"{q_prefix}{test_sentence}\n{a_prefix}"
    if a_prefix[-1] == " ":  # GPT-3 does not need a trailing space
        prompt = prompt[:-1]
    
    return prompt


def get_responses(prompts: List[str], echo: bool = False) -> List[Any]:
    """
    Get responses from the language model for given prompts.

    Args:
        prompts (List[str]): List of prompts to send to the model
        echo (bool, optional): If True, returns logprobs for top 10 tokens. Defaults to False.

    Returns:
        List[Any]: List of model responses, each containing logprobs and other completion data
    """

    # Set OpenAI according to the instruction file in README
    client = openai.OpenAI(
        api_key=...,  # please use .env file to store your key
        base_url=...,
    )
    # Get responses
    responses = []
    max_tokens = 0 if echo else 1
    for prompt in tqdm(prompts, desc="Get response"):
        response = client.completions.create(
            model="davinci-002",
            prompt=prompt,
            max_tokens=max_tokens,
            stop='\n',  # stop at the first newline
            logprobs=1,  # return log prob of top-1 token
            echo=echo  # whether to also return the input
        )
        responses.append(response.choices[0])

    return responses


def get_label_probs(
    responses: List[Any], 
    few_shot_sentences: List[str], 
    few_shot_labels: List[str], 
    test_sentences: List[str],
    q_prefix: str,
    a_prefix: str
) -> np.ndarray:
    """
    Calculate label probabilities from model responses.

    Args:
        responses (List[Any]): Model responses containing logprobs
        few_shot_sentences (List[str]): Example sentences used in prompts
        few_shot_labels (List[str]): Labels for example sentences
        test_sentences (List[str]): Test sentences to analyze
        q_prefix (str): Prefix of the review
        a_prefix (str): Prefix of the sentiment

    Returns:
        np.ndarray: Array of label probabilities (not normalized)
    """

    label_dict = {0: "Negative", 1: "Positive"}
    num_labels = len(label_dict)
    all_label_probs = []
    all_missing_positions = []

    # Initial probabilities from model responses
    for i, response in enumerate(tqdm(responses, desc="Get initial prob")):
        top_logprobs = ...
        label_probs = ...

        for j, label in label_dict.items():
            if a_prefix[-1] == " ":
                label = " " + label  # add space to match the format

            if label in top_logprobs:
                label_probs[j] += ...
            else:
                # add to missing positions
                ...

        all_label_probs.append(label_probs)
    all_label_probs = np.array(all_label_probs)

    # Fill in missing positions
    all_additional_prompts = []
    for i, j in all_missing_positions:
        prompt = ...

        if a_prefix[-1] == " ":
            prompt += " " + label

        all_additional_prompts.append(prompt)

    additional_responses = get_responses(all_additional_prompts, echo=True)

    for idx, (i, j) in enumerate(all_missing_positions):
        prob = ...
        all_label_probs[i][j] = prob

    return all_label_probs  # not normalized


def calibrate(
    content_free_input: str, 
    few_shot_sentences: List[str], 
    few_shot_labels: List[str], 
    q_prefix: str,
    a_prefix: str
) -> np.ndarray:
    """
    Calculate calibration vector using content-free input.

    Args:
        content_free_input (str): Content-free input text (e.g., "N/A")
        few_shot_sentences (List[str]): Example sentences used in prompts
        few_shot_labels (List[str]): Labels for example sentences
        q_prefix (str): Prefix of the review
        a_prefix (str): Prefix of the sentiment

    Returns:
        np.ndarray: Calibration vector (normalized probabilities)
    """

    label_dict = {0: "Negative", 1: "Positive"}
    num_labels = len(label_dict)

    prompt = ...
    p_y = [0] * num_labels

    for i, answer in label_dict.items():
        if a_prefix[-1] == " ":  # to match the prompt format
            key = " "+answer
        else:
            key = answer

        response = get_responses(prompts=[prompt+key], echo=True)[0]
        p_y[i] = ...

    p_y = ...

    return p_y


def eval_accuracy(all_label_probs: np.ndarray, test_labels: List[int], p_cf: Optional[np.ndarray] = None) -> float:
    """
    Evaluate classification accuracy with optional calibration.

    Args:
        all_label_probs (np.ndarray): Array of label probabilities for each test sentence
        test_labels (List[int]): True labels for test sentences
        p_cf (Optional[np.ndarray], optional): Calibration vector. Defaults to None.

    Returns:
        float: Classification accuracy (between 0 and 1)
    """

    # We use diagonal matrix here as the paper mentions it's better than the identity matrix for classification
    num_labels = ...

    if p_cf is None:
        W = ...
        b = ...
    else:
        W = ...
        b = ...

    corrects = []
    for prob, label in zip(all_label_probs, test_labels):
        ...

    accuracy = np.mean(corrects)
    return accuracy