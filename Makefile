include .env

clean_docker:
	docker container prune -f
	docker image prune -f

setup_llama:
	curl -X POST http://localhost:8080/setup_triton_server \
		-H "Content-Type: application/json" \
		-d '{"model_name": "llama", "base_model_repo_id": "meta-llama/Llama-3.2-1B", "hf_read_token": "$(HF_TOKEN)", "tokenizer_type": "llama", "precision_type": "float16", "max_batch_size": 8, "max_input_len": 128, "max_output_len": 64}'

setup_tiny:
	curl -X POST http://localhost:8080/setup_triton_server \
		-H "Content-Type: application/json" \
		-d '{"model_name": "tiny", "base_model_repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "hf_read_token": "$(HF_TOKEN)", "tokenizer_type": "llama", "precision_type": "float16", "max_batch_size": 2, "max_input_len": 128, "max_output_len": 64}'

run:
	curl -X POST http://127.0.0.1:8080/run_triton_server \
		-H "Content-Type: application/json" \
		-d '{"http_port": 8000, "grpc_port": 8001, "metrics_port": 8002}'

inference:
	curl -X POST localhost:8000/v2/models/llama/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": ["intelligence", "model"], "stop_words": ["focuses", "learn"], "pad_id": 2, "end_id": 2}'