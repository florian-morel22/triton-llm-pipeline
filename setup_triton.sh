#!/bin/bash

# Reads the named argument values
while [ $# -gt 0 ]; do
    case "$1" in
        --base_model_repo_id) base_model_repo_id="$2"; shift;;
        --model_name) model_name="$2"; shift;;
        --hf_read_token) hf_read_token="$2"; shift;;
        --tokenizer_type) tokenizer_type="$2"; shift;;
        --precision_type) precision="$2"; shift;;
        --max_batch_size) max_batch_size="$2"; shift;;
        --max_input_len) max_input_len="$2"; shift;;
        --max_output_len) max_output_len="$2"; shift;;
        --max_context_len) max_context_len="$2"; shift;;
        --max_beam_width) max_beam_width="$2"; shift;;
        --alpha) alpha="$2"; shift;;
        --logits_datatype) logits_datatype="$2"; shift;;
        --decoupled_mode) decoupled_mode="$2"; shift;;
        --max_queue_size) max_queue_size="$2"; shift;;
        --max_queue_delay_microseconds) max_queue_delay_microseconds="$2"; shift;;
        --instance_count) instance_count="$2"; shift;;
        --skip_special_tokens) skip_special_tokens="$2"; shift;;
        --triton_backend) triton_backend="$2"; shift;;
        --batching_strategy) batching_strategy="$2"; shift;;
        --kv_cache_free_gpu_mem_fraction) kv_cache_free_gpu_mem_fraction="$2"; shift;;
        --exclude_input_in_output) exclude_input_in_output="$2"; shift;;
        --enable_kv_cache_reuse) enable_kv_cache_reuse="$2"; shift;;
        --enable_chunked_context) enable_chunked_context="$2"; shift;;
        --encoder_input_features_data_type) encoder_input_features_data_type="$2"; shift;;
        --) shift;;
    esac
    shift
done



###################

converted_weights_dir="$PWD/tmp/trt_engines/1-gpu/"
converted_model_dir="$PWD/tmp/trt_engines/compiled-model/"
tokenizer_dir="$PWD/tokenizers/$tokenizer_type/"
mkdir -p "$tokenizer_dir"

echo "cleaning up model directories"
rm -rf "$converted_weights_dir"
rm -rf "$converted_model_dir"

echo "rebuilding model directories"
mkdir -p "$converted_weights_dir"
mkdir -p "$converted_model_dir"

source_model_dir="$PWD/tmp/hf_models/$base_model_repo_id"
model_dir="$PWD/models/"

python download_model.py "$source_model_dir" "$base_model_repo_id" "$hf_read_token"

###################

#### TO UPDATE to make it models agnostic. See new version of tensorrt llm with examples/core/ ####

echo "converting weights"
python tensorrtllm_backend/tensorrt_llm/examples/llama/convert_checkpoint.py --model_dir "$source_model_dir" \
                                                         --output_dir "$converted_weights_dir" \
                                                         --dtype "$precision"
echo "weights converted"

###################

max_num_tokens_float="$(echo "scale=0; ($max_batch_size * $max_input_len * $alpha) + ($max_batch_size * $max_beam_width * (1 - $alpha))" | bc)"
max_num_tokens="$(printf "%.0f" "$max_num_tokens_float")"

echo "max_num_tokens: $max_num_tokens"

trtllm-build --checkpoint_dir "$converted_weights_dir" \
            --output_dir "$converted_model_dir" \
            --gemm_plugin "$precision" \
            --max_input_len "$max_input_len" \
            --max_num_tokens  "$max_num_tokens" \
            --gpt_attention_plugin "$precision" \
            --remove_input_padding enable \
            --paged_kv_cache enable \
            --max_batch_size "$max_batch_size" \
            --use_fused_mlp enable \
            --context_fmha enable \
            --use_paged_context_fmha enable \
            # --max_output_len "$max_output_len" \
            # --use_context_fmha_for_generation disable \


echo "model compiled"

###################

echo "copying model files"
mkdir -p "$model_dir"
cp -r "$PWD/tensorrtllm_backend/all_models/inflight_batcher_llm/tensorrt_llm" "$model_dir/"
cp -r "$PWD/tensorrtllm_backend/all_models/inflight_batcher_llm/ensemble" "$model_dir/"
cp -r "$PWD/tensorrtllm_backend/all_models/inflight_batcher_llm/preprocessing" "$model_dir/"
cp -r "$PWD/tensorrtllm_backend/all_models/inflight_batcher_llm/postprocessing" "$model_dir/"
cp -r "$converted_model_dir." "$model_dir/tensorrt_llm/1/"

echo "copying tokenizer"
cp "$source_model_dir/tokenizer.json" "$tokenizer_dir"
cp "$source_model_dir/tokenizer_config.json" "$tokenizer_dir"

echo "configuring tokenizer"
# Configuring preprocessing
python "$PWD/tensorrtllm_backend/tools/fill_template.py" --in_place \
    "$model_dir/preprocessing/config.pbtxt" \
    tokenizer_dir:"$tokenizer_dir",triton_max_batch_size:"$max_batch_size",preprocessing_instance_count:"$instance_count",max_queue_size:"$max_queue_size"

# Configuring postprocessig
python "$PWD/tensorrtllm_backend/tools/fill_template.py" --in_place \
    "$model_dir/postprocessing/config.pbtxt" \
    tokenizer_dir:"$tokenizer_dir",triton_max_batch_size:"$max_batch_size",postprocessing_instance_count:"$instance_count",skip_special_tokens:"$skip_special_tokens"

echo "configuring model"
# Configuring ensemble (chained model: preprocessing > tensorrt_llm > postprocessing)
python3 "$PWD/tensorrtllm_backend/tools/fill_template.py" --in_place \
    "$model_dir/ensemble/config.pbtxt" \
    triton_max_batch_size:"$max_batch_size",logits_datatype:"$logits_datatype"

# Configuring tensorrt_llm
python3 "$PWD/tensorrtllm_backend/tools/fill_template.py" --in_place \
    "$model_dir/tensorrt_llm/config.pbtxt" \
    decoupled_mode:"$decoupled_mode",engine_dir:"$converted_model_dir",triton_max_batch_size:"$max_batch_size",max_beam_width:1,max_attention_window_size:"$max_context_len",kv_cache_free_gpu_mem_fraction:"$kv_cache_free_gpu_mem_fraction",exclude_input_in_output:"$exclude_input_in_output",enable_kv_cache_reuse:"$enable_kv_cache_reuse",batching_strategy:"$batching_strategy",max_queue_delay_microseconds:"$max_queue_delay_microseconds",enable_chunked_context:"$enable_chunked_context",logits_datatype:"$logits_datatype",encoder_input_features_data_type:"$encoder_input_features_data_type",triton_backend:"$triton_backend",max_queue_size:"$max_queue_size"



# echo "cleaning up temp directories"
# rm -rf "$source_model_dir"
# rm -rf "$converted_weights_dir"
# rm -rf "$converted_model_dir"