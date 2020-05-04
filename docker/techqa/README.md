# TechQA Submission Docker

This page contains instructions about how to make a docker submission to the [TechQA leaderboard](https://leaderboard.techqa.us-east.containers.appdomain.cloud)

This docker image runs a single model for prediction and can decode on GPUs. Note that this Docker setup has only been tested with BERT models (not RoBERTa etc.), so other models may require modification of these files to work. 

Note: All submissions to the leaderboard run without any network access. Please make sure how have downloaded all external resources during image construction.

## Building Image

1) Move the trained model, vocabulary and tokenizer files into the folder `models`. You may need to create the `models` directory if this is your first time building the image. A sample `models` folder with a BERT model would have the following files: `config.json, pytorch_model.bin, special_tokens_map.json, tokenizer_config.json, vocab.txt`

2) Update the `MODEL_TYPE` environment variable in the [Dockerfile](./Dockerfile) if needed.
3) Update the [submission script](./submission.sh) with any command line parameters that need to be changed for
  your submission (e.g. `threshold`, `add_doc_title`, etc.)
4) Run from project's top level directory:
```docker build -f ./docker/techqa/Dockerfile -t techqa:<image_name_here> .```

## Running Image

To test the image on the validation set, run:

```
nvidia-docker run --rm -e CUDA_VISIBLE_DEVICES=0 --network none -v <PATH TO folder containing validation questions and documents>:/var/spool/TechQA/input/:ro -v <PATH TO OUTPUT FOLDER>:/var/spool/TechQA/output -e DEV_INPUT_CORPUS=/var/spool/TechQA/input/documents.json techqa:<image_name_here>
```

## Notes

- If you want to see how the image performs without network access,
 add `--network none` to the `docker run` command.
- Replace `CUDA_VISIBLE_DEVICES` with the comma-separated GPU ids to run on,
 or omit to run on all GPUS.
- Remove `--rm` to prevent container from being removed after stopping.
- You can override the input query file with `-e DEV_QUERY_FILE=/var/spool/TechQA/input/validation_questions.json`,
 the same can be done for corpus file `DEV_INPUT_CORPUS` and output file `DEV_OUTPUT_FILE` (defaults in the [Dockerfile](./Dockerfile)).
- If you have Docker 19.03 or later, `nvidia-docker` is no longer needed as NVIDIA GPUs are natively supported in Docker.
  See the [documentation](https://github.com/NVIDIA/nvidia-docker#quickstart) for more information.
## Submitting to TechQA leaderboard

1) Build Image
2) Test Image
3) Push image to your docker registry
4) Go to the submission site: https://leaderboard.techqa.us-east.containers.appdomain.cloud
5) Create an account if you don't have on already and login.
6) Click on the `Create Submission` button to see the submission form.

