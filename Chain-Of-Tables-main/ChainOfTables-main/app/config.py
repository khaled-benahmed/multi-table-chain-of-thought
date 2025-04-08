config = {
    "inference_server_url": "http://127.0.0.1:7000", 
    "use_possible_next_operations": True,
    "shuffle_possible_next_operations": False,
    "guidance_type": "json",
    "log_llm_output": True,
    "MAX_FAILED_REPETITIONS": 3,
    "MAX_CHAIN_LENGTH": 8,  # Consider increasing this for multi-table operations
    "enable_multi_table": True,  # New flag to enable multi-table operations
}