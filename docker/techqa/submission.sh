#!/usr/bin/env bash
set -xeo pipefail
. /opt/conda/etc/profile.d/conda.sh
conda activate base

set -u  # This is after conda setup as conda setup has unbound vars which trigger it

if [[ "$#" -ne 3 ]]; then
    echo "Usage: bash submission.sh QUERY_FILE INPUT_CORPUS OUTPUT_FILE"
fi

QUERY_FILE="${1}"
CORPUS_FILE="${2}"
OUTPUT_FILE="${3}"

SCRIPT_LOCATION="run_techqa.py"

# Input data in readonly volume, need to symlink into container for featurization to work
function symlink_and_return_name() {
    SOURCE="$(realpath ${1})"
    TARGET_DIR="$(realpath ${2})"
    TARGET="${TARGET_DIR}/$(basename ${SOURCE})"
    ln -s "${SOURCE}" "${TARGET}"
    echo "${TARGET}"
}

SYMLINKED_QUERY_PATH=$(symlink_and_return_name "${QUERY_FILE}" "${SYMLINK_DIR}")
SYMLINKED_CORPUS_PATH=$(symlink_and_return_name "${CORPUS_FILE}" "${SYMLINK_DIR}")

python ${SCRIPT_LOCATION} \
--model_type "${MODEL_TYPE}" --model_name_or_path "${MODEL_WEIGHTS}" --do_lower_case \
--tokenizer_name "${MODEL_WEIGHTS}" \
--fp16 --do_eval --predict_file ${SYMLINKED_QUERY_PATH} --input_corpus_file ${SYMLINKED_CORPUS_PATH} \
--overwrite_output_dir --output_dir ${TMP_OUTPUT_DIR} --add_doc_title_to_passage --threshold 23.912109375

# Copy predictions
cp ${TMP_OUTPUT_DIR}/predictions.json ${OUTPUT_FILE}