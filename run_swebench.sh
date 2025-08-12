PREDS_PATH="Qwen__Qwen3-14B_preds_2.jsonl"
RUN_ID="Qwen__Qwen3-14B_preds_2"

python -m swebench.harness.run_evaluation \
    --dataset_name rasdani/SWE-bench_Verified_oracle-parsed_commits_32k_100 \
    --predictions_path $PREDS_PATH \
    --run_id $RUN_ID \
    --namespace none