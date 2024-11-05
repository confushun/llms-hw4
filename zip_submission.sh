#!/bin/bash

rm -rf submission
mkdir submission
cp -r src setup.py requirements.txt submission/
cd submission
zip -qr ../submission.zip . -x src/bias/__pycache__/**\* src/olmo/__pycache__/**\* src/pytest_utils/__pycache__/**\* src/mnli/__pycache__/**\* src/cmu_llms_hw4.egg-info/**\*