# Building a Baseline Model for the TechQA dataset

This repo contains code to train and run a baseline model against the TechQA dataset released in the ACL 2020 paper: [The TechQA dataset](https://arxiv.org/abs/1911.02984)

This dataset has a leaderboard task which can be found here: http://ibm.biz/Tech_QA

Please refer [here](./docker/techqa/README.md) for details about how to make a submission to the leaderboard.

## Installation


After cloning this repo, install dependencies using 
```
pip install -r requirements.txt
```

If you want to run with `fp16`, you need to install [Apex]( https://github.com/NVIDIA/apex.git)

## Training a model

In order to train a model on TechQA, use the script below. 

Note: Since TechQA is smaller dataset, it is better to start with a model that is already trained on a bigger QA dataset. Here, we start with BERT-Large trained on Squad.

```
python run_techqa.py \
    --model_type bert \
    --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
    --do_lower_case \
    --learning_rate 5.5e-6 \ 
    --do_train \
    --num_train_epochs 20 \
    --train_file <PATH TO training_Q_A.json> \
    --do_eval \
    --predict_file <PATH TO dev_Q_A.json> \
    --input_corpus_file <PATH TO training_dev_technotes.json> \
    --overwrite_output_dir \
    --output_dir <PATH TO OUTPUT FOLDER> \ 
    --add_doc_title_to_passage 
```

You can add the `--fp16` flag if you have apex installed.

To evaluate a model, you can run:

```
python run_techqa.py \
    --model_type bert \
    --model_name_or_path <PATH TO TRAINED MODEL FOLDER> \
    --do_lower_case \
    --do_eval \
    --predict_file <PATH TO dev_Q_A.json> \
    --input_corpus_file <PATH TO training_dev_technotes.json> \
    --overwrite_output_dir \
    --output_dir <PATH TO OUTPUT FOLDER> \ 
    --add_doc_title_to_passage 
```


## Contact

For help or issues, please submit a GitHub issue.

For direct communication, please contact Avi Sil (avi@us.ibm.com).

