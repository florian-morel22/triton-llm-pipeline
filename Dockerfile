FROM nvcr.io/nvidia/tritonserver:25.03-trtllm-python-py3

WORKDIR /app

RUN apt update && apt install -y bc

RUN git clone https://github.com/triton-inference-server/tensorrtllm_backend.git /app/tensorrtllm_backend -b v0.18.2
RUN cd /app/tensorrtllm_backend && git submodule update --init --recursive tensorrt_llm


COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY main.py /app/main.py
COPY download_model.py /app/download_model.py
COPY setup_triton.sh /app/setup_triton.sh
COPY config_template /app/config_template

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]