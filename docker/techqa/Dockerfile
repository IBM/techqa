# Start with nvidia pytorch image (ensures working pytorch + CUDA)
FROM nvcr.io/nvidia/pytorch:19.04-py3

# Setup tmp workdirs
ENV SYMLINK_DIR /symlinked_data
ENV TMP_OUTPUT_DIR /tmp_outputs
RUN mkdir "${SYMLINK_DIR}"
RUN mkdir "${TMP_OUTPUT_DIR}"
ENV PYTHONPATH "."

# Copy model code and install
ADD . /techqa_model
WORKDIR /techqa_model

# Model Location
ENV MODEL_WEIGHTS "models/"
ENV MODEL_TYPE "bert"

# Data Files
ENV DEV_QUERY_FILE /var/spool/TechQA/input/questions.json
ENV DEV_INPUT_CORPUS /var/spool/TechQA/input/documents.json
ENV DEV_OUTPUT_FILE /var/spool/TechQA/output/sys_output.json

# Install python packages
RUN conda env update -n base -f docker/techqa/environment-techqa.yml
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo "conda activate base" >> ~/.bashrc

CMD bash docker/techqa/submission.sh "${DEV_QUERY_FILE}" "${DEV_INPUT_CORPUS}" "${DEV_OUTPUT_FILE}"