import time
import requests
import subprocess
import threading

from typing import IO
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field

TRITON_HEALTH_URL = "http://localhost:8000/v2/health/ready"  # Adjust if needed
MAX_WAIT_SECONDS = 20

app = FastAPI()

def stream_output(stream: IO[bytes], flag: str):
    for line in iter(stream.readline, ''):
        print(f"{flag} : {line.strip()}")



class SetupParams(BaseModel):
    model_name: str = Field(..., description="Name of the model to be used. e.g., 'llama'")
    hf_read_token: str = Field(..., description="Hugging Face read token.")
    base_model_repo_id: str = Field(..., description="Hugging Face repo ID of the base model (e.g., 'meta-llama/Llama-2-7b-hf')")
    tokenizer_type: str = Field(..., description="Type or name of the tokenizer to be used. e.g., 'llama'")
    precision_type: str = Field(..., description="Supported precision types: 'float16', 'bfloat16, 'float32'")
    max_batch_size: int = Field(..., description="max_batch_size is optional, and will default to 1 if not specified")
    max_input_len: int = Field(..., description="")
    max_output_len: int = Field(..., description="")
    max_context_len: int = Field(8192, description="")
    max_beam_width: int = Field(1, description="")
    alpha: float = Field(0.2, description="")

    logits_datatype: str = Field("TYPE_FP32", description="")
    decoupled_mode: str = Field("false", description="Whether to use decoupled mode. Must be set to true for requests setting the stream tensor to true. Supported decoupled modes: 'true', 'false'")
    max_queue_size: int = Field(1, description="The maximum number of requests allowed in the TRT-LLM queue before rejecting new requests.")
    max_queue_delay_microseconds: int = Field(0, description="The maximum queue delay in microseconds. Setting this parameter to a value greater than 0 can improve the chances that two requests arriving within max_queue_delay_microseconds will be scheduled in the same TRT-LLM iteration.")
    instance_count: int = Field(1, description="")

    skip_special_tokens: str = Field("true", description="")

    triton_backend: str = Field("tensorrtllm", description="Supported backends: 'tensorrtllm', 'python'")
    batching_strategy: str = Field("inflight_fused_batching", description="The batching strategy to use. Set to inflight_fused_batching when enabling in-flight batching support. To disable in-flight batching, set to V1. Supported batching strategies: 'inflight_fused_batching', 'V1'")

    kv_cache_free_gpu_mem_fraction: float = Field(0.9, description="Set to a number between 0 and 1 to indicate the maximum fraction of GPU memory (after loading the model) that may be used for KV cache. (default=0.9)")
    exclude_input_in_output: str = Field("false", description="Set to true to only return completion tokens in a response. Set to false to return the prompt tokens concatenated with the generated tokens. (default=false)")
    enable_kv_cache_reuse: str = Field("false", description="Set to true to reuse previously computed KV cache values (e.g. for system prompt)")
    enable_chunked_context: str = Field("false", description="Set to true to enable context chunking. (default=false)")
    encoder_input_features_data_type: str = Field("TYPE_FP16", description="The dtype for the input tensor encoder_input_features. For the mllama model, this must be TYPE_BF16. For other models like whisper, this is TYPE_FP16.")



@app.post("/setup_triton_server")
def setup_triton_server(params: SetupParams):

    cmd = [
        "sh", "setup_triton.sh",
        "--model_name", params.model_name,
        "--hf_read_token", params.hf_read_token,
        "--base_model_repo_id", params.base_model_repo_id,
        "--tokenizer_type", params.tokenizer_type,
        "--precision_type", params.precision_type,
        "--max_batch_size", str(params.max_batch_size),
        "--max_input_len", str(params.max_input_len),
        "--max_output_len", str(params.max_output_len),
        "--max_context_len", str(params.max_context_len),
        "--max_beam_width", str(params.max_beam_width),
        "--alpha", str(params.alpha),

        "--logits_datatype", params.logits_datatype,
        "--decoupled_mode", params.decoupled_mode,
        "--max_queue_size", str(params.max_queue_size),
        "--max_queue_delay_microseconds", str(params.max_queue_delay_microseconds),
        "--instance_count", str(params.instance_count),
        "--skip_special_tokens", params.skip_special_tokens,
        "--triton_backend", params.triton_backend,
        "--batching_strategy", params.batching_strategy,
        "--kv_cache_free_gpu_mem_fraction", str(params.kv_cache_free_gpu_mem_fraction),
        "--exclude_input_in_output", params.exclude_input_in_output,
        "--enable_kv_cache_reuse", params.enable_kv_cache_reuse,
        "--enable_chunked_context", params.enable_chunked_context,
        "--encoder_input_features_data_type", params.encoder_input_features_data_type
    ]

    try:

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        thread = threading.Thread(target=stream_output, args=(process.stdout, "[setup-triton-server]"))
        thread.daemon = True 
        thread.start()

        process.wait()

        if process.returncode == 0:
            return {
                "status": "success",
                "message": "Triton server setup completed successfully."
            }
        else:
            return {
                "status": "error",
                "message": f"Triton server setup failed with return code {process.returncode}."
            }

    except Exception as e:
        return {
            "status": "exception",
            "message": str(e)
        }

        
class RunParams(BaseModel):
    world_size: int = Field(1, description="world size, only support tensor parallelism now.")
    http_port: int = Field(8000, description="http port.")
    grpc_port: int = Field(8001, description="grpc port.")
    metrics_port: int = Field(8002, description="metrics port.")


@app.post("/run_triton_server")
def run_triton_server(params: RunParams):

    cmd = [
        "python3",
        "tensorrtllm_backend/scripts/launch_triton_server.py",
        "--world_size", str(params.world_size),
        "--model_repo", f"models/",
        "--http_port", str(params.http_port),
        "--grpc_port", str(params.grpc_port),
        "--metrics_port", str(params.metrics_port),
    ]

    try:

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        thread = threading.Thread(target=stream_output, args=(process.stdout, "[run-triton-server]"))
        thread.daemon = True 
        thread.start()

        start_time = time.time()
        while time.time() - start_time < MAX_WAIT_SECONDS:
            try:
                r = requests.get(TRITON_HEALTH_URL)
                if r.status_code == 200:
                    return {"status": "success", "message": "Triton server is up."}
            except Exception:
                print("Triton server not ready yet, waiting...")
            time.sleep(1)

        return {
            "status": "timeout",
            "message": f"Triton server did not respond within {MAX_WAIT_SECONDS} seconds."
        }

    except Exception as e:
        return {
            "status": "exception",
            "message": str(e)
        }


@app.post("/stop_triton_server")
def stop_triton_server():
   """ stop the triton server """
   
   pass


