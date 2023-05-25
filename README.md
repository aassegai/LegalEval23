# LegalEval 2023
## Description
This repository is aimed at solving subtask A of LegalEval task during SemEval 23. The main objective was to label the sentences of a segmented document as a rhetorical roles. My solution is an Indian Legal BERT with a linear layer on top. Model achieves 0.83 weighted F-1 score on a test set. Main solution is in [this notebook](/LegalEval_Task_A.ipynb), some experiments [here](/ModelTesting.ipynb)
## Repo contents
### [utils](/utils.py)
This module contains technical scripts for output representation
### [data](/data/raw)
Competition dataset inside
###[src/datasets](/src/datasets)
Contains modules for dataset construction
###[src/model](/src/model)
Contains custom trainer module and a folder for custom model architectures.
###[src/preprocessing](/src/preprocessing)
Contains data preprocessing scripts 
