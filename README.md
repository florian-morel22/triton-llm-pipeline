# triton-llm-pipeline

Automatization of the deployment of a triton server for llm inference.

For now, only llama models can be deployed.

You can build the docker image using the Dockerfile in the repo, or directly pull the image from :
```
florianmorel22/triton-llm-pipeline:0.0.3
```

In the ```Makefile```, you can find the possible requests to send to the servers (**Uvicorn** for triton server deployement, and **Triton** for inference)