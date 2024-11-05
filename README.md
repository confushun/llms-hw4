# 11-667 Homework 4: Comparing Models and Mitigating Bias (ver 2024.1.1)

## Setting up

### AWS
If you do not already have access to GPUs, you may need an AWS virtual
  machine for model training.
[Here are the instructions for setting that up.](https://docs.google.com/presentation/d/1Tw_klO84R9G7CZ3cINAKgy4BfdNm-8dlnRXSBIVD_3A/edit?usp=sharing) 

### Python environment
1. Install conda: `bash setup-conda.sh && source ~/.bashrc`
2. Create conda environment:
   If you run into error like `UnavailableInvalidChannel: HTTP 403 FORBIDDEN for channel <some channel>` on your EC2 instance, you can solve it by running `conda config --remove channels <some channel>`, and make sure you have the default channel by running `conda config --add channels defaults`.
```bash
conda create -n cmu-llms-hw4 python=3.11
conda activate cmu-llms-hw4
pip install -r requirements.txt
pip install -e .
```

*Note: To ensure that you have set up the Python environment correctly, you should run
`pytest tests/test_env.py` and confirm that the test case passes.*

## Classification bias calibration
In this problem, you will use the GPT-3 model with OpenAI API. This is done by using LiteLLM API proxy, and we have sent out your own key earlier this semester. Please set your `LITELLM_API_KEY` in `.evn`. For more details about how to make an API call, you may refer to this [instruction](https://docs.google.com/document/d/1RpNCBoBVqSPvo4tD5LCjxkTT2UEo4_KJPfqYEl0Fs-I/edit?tab=t.0). After completing `utils.py`, you may run the algorithm in `classification.py`.

## Testing
You can test your solutions by running `pytest` in the project directory. Initially all test cases will fail, and you should check your implementation against the test cases as you are working through the assignment.

## Code submission
1. Run `zip_submission.sh`
2. Upload the generated `submission.zip` to Gradescope


## Acknowledgement
This code contains adaptations from [few-shot-learning](https://github.com/tonyzhaozh/few-shot-learning)([license](copyright/few-shot-learning)).