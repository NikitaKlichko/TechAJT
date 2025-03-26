# Linguistic Acceptability Model for Technical Texts  

This repository contains experimental research on model training, as well as the final model for assessing linguistic acceptability in technical texts. The experiments explore various approaches to evaluating whether a given text meets linguistic and stylistic standards, particularly in specialized technical domains. The resulting model is designed to automatically determine the grammatical correctness, coherence, and overall linguistic quality of technical writing. 

## Requirements  

To run the code and use the model, ensure you have the following dependencies installed:  

```python
pip install -r requirements.txt
```

# Model Usage
To use the trained model, follow these steps:

1. Install dependencies (see Requirements).

2. Load the model:
See the example script [predict.py](./predict.py). You will need to specify:
    * Path to the trained model, init model and tokenizer (see [model files](https://drive.google.com/drive/folders/1Qyh-HxzcyGpZPDDPbUphi9UDkuGuBBEc?usp=sharing))
    * Input text for evaluation

# Dataset
The dataset consists of 1,000 annotated samples extracted from technical reports and final qualification theses (Bachelor's and Master's) written by students. 
For details, explore the dataset files in [data](/data).
