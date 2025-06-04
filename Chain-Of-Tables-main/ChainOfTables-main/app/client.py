import ast
import asyncio
import json
import logging
import os
import time
from datetime import datetime

import pandas as pd
import requests
from datasets import load_dataset
from tqdm import tqdm

from agent import Agent
from config import config
from eval import Evaluator

TAGGED_DATASET_PATH = os.path.join('.', 'tagged', 'data')



def compile_natural_table(nt):
    """
    Compile either a single table or multiple tables from the input.
    """
    # Check if it's a multi-table format
    if "tables" in nt:
        tables = {}
        for table_info in nt["tables"]:
            table_name = table_info["name"]
            header = table_info["header"]
            rows = table_info["rows"]
            tables[table_name] = pd.DataFrame(rows, columns=header)
        return nt["question"], tables, nt["answers"]
    
    # Original single-table format
    header = nt["table"]["header"]
    rows = nt["table"]["rows"]
    df = pd.DataFrame(rows, columns=header)
    return nt["question"], df, nt["answers"]


def setup_instance_logger(logs_subdirectory, id):
    logger = logging.getLogger(f"instance_{id}")
    logger.setLevel(logging.DEBUG)
    log_filename = f"{logs_subdirectory}/log_test_{id}.log"

    # Check if the logger already has handlers
    if not logger.handlers:
        # Create a file handler for the logger
        fh = logging.FileHandler(log_filename)
        fh.setLevel(logging.WARNING)

        # Create a formatter and set it for the handler
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        fh.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(fh)

    return logger, log_filename


# Asynchronous function to process a single instance
async def process_instance(instance, evaluator, logs_subdirectory, semaphore):
    async with semaphore:

        id = instance["id"]
        logger, log_filename = setup_instance_logger(logs_subdirectory, id)
        try:
            question, table, answers = compile_natural_table(instance)
            start_time = time.time()
            generated_answer, chain, metadata = await Agent(logger).chain_of_table(
                table, question
            )
            metadata["processing_time"] = round(time.time() - start_time, 3)

            evaluation_result = await evaluate_answer(evaluator, question, generated_answer, logger)
            metadata["evaluation_result"] = evaluation_result
            
            metadata["chain_length"] = chain.length()
            
            return {
                "id": id,
                "target": answers,
                "predicted": generated_answer,
                "actions": str(chain),
                "log_file": log_filename,
                **metadata,
            }
        except Exception as e:
            logger.error(f"Unexpected error in instance {id}: {e}")
            return None

async def evaluate_answer(evaluator, question, generated_answer, logger):
    try:
        if isinstance(generated_answer, str):
            answer_list = ast.literal_eval(generated_answer)
            answer_list = [str(element) for element in answer_list]
        else:
            answer_list = generated_answer

        evaluation_result = evaluator.evaluate(question, answer_list)
        return evaluation_result
    except Exception as e:
        logger.warning(f"Evaluation failed: {e}")
        return None
    


async def async_process_dataset(dataset, evaluator, log_directory, max_concurrent_tasks=8):
    # Use asyncio to process the instances asynchronously
    results = []
    logs_subdirectory = os.path.join(log_directory, "logs")
    os.makedirs(logs_subdirectory, exist_ok=True)
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    tasks = [
        asyncio.create_task(process_instance(instance, evaluator, logs_subdirectory, semaphore))
        for instance in dataset["test"]
    ]

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await coro
        if result is not None:
            results.append(result)

        # Save results in batches of 5
        if len(results) % 5 == 0:
            pd.DataFrame(results).to_csv(
                f"{log_directory}/experiment.csv", index=False
            )

    # Ensure to write remaining data if total iterations are not a multiple of 5
    if len(results) % 5 != 0:
        pd.DataFrame(results).to_csv(f"{log_directory}/experiment.csv", index=False)

def save_config(log_directory):
    experiment_config_path = os.path.join(log_directory, "experiment_config.json")
    with open(experiment_config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    try:
        server_config = requests.get(config.get("inference_server_url") + "/info").json()
        server_config_path = os.path.join(log_directory, "server_config.json")
        with open(server_config_path, "w") as f:
            json.dump(server_config, f, indent=4)
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch server config: {e}")

async def main():
    dataset = load_dataset("Stanford/wikitablequestions")
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_directory = f"Experiments/{current_time}"
    os.makedirs(log_directory, exist_ok=True)

    logs_subdirectory = os.path.join(log_directory, "logs")
    os.makedirs(logs_subdirectory, exist_ok=True)

    save_config(log_directory)

    evaluator = Evaluator(TAGGED_DATASET_PATH)

    await async_process_dataset(
        dataset, evaluator, log_directory, max_concurrent_tasks=16  # Adjust as needed
    )


if __name__ == "__main__":
    asyncio.run(main())
