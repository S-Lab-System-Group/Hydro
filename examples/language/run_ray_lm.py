import math
import argparse
from itertools import chain

import torch
from torch.utils.data import DataLoader

from transformers.data.data_collator import torch_default_data_collator
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config
from datasets import load_dataset
from accelerate import Accelerator

import ray
import ray.train.torch as ht
from ray.air import session
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray import tune
from ray.tune.tuner import Tuner
from ray.tune.tune_config import TuneConfig


SEARCH_SPACE = {
    "lr": tune.qloguniform(1e-5, 0.1, 1e-5),
    "gamma": tune.quniform(0.01, 0.9, 0.01),
}


def get_dataset(accelerator, config):
    datasets = load_dataset(config.get("dataset_path"), config.get("dataset_name"))
    tokenizer_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    block_size = tokenizer.model_max_length
    # block_size = 128

    # Preprocessing the datasets
    column_names = datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with accelerator.main_process_first():
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=None,
            remove_columns=column_names,
        )

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            batch_size=1000,
            num_proc=None,
        )

    return lm_datasets["train"], lm_datasets["validation"]


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def train_func(config):
    accelerator = Accelerator()

    train_set, val_set = get_dataset(accelerator, config)

    assert "gpt2" in config.get("model")
    model_config = GPT2Config(n_layer=2)
    model = AutoModelForCausalLM.from_config(model_config)
    model = ht.prepare_model(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 5e-5))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=config.get("gamma", 0.2), weight_decay=0.001)

    worker_batch_size = config["batch_size"] // session.get_world_size()

    train_loader = DataLoader(
        train_set,
        batch_size=worker_batch_size,
        collate_fn=torch_default_data_collator,
        num_workers=1,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=worker_batch_size,
        collate_fn=torch_default_data_collator,
        num_workers=1,
        pin_memory=True,
    )

    train_loader = ht.prepare_data_loader(train_loader)
    val_loader = ht.prepare_data_loader(val_loader)

    train_iterator = iter(cyclic_iter(train_loader))

    for i in range(config.get("max_iter")):
        # Train model
        model.train()
        batch = next(train_iterator)
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs["loss"]
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        # Eval model
        if i % config.get("eval_interval") == 0 and i > 0:
            # Eval model
            model.eval()
            losses = []
            for step, batch in enumerate(val_loader):
                with torch.no_grad():
                    outputs = model(**batch)

                loss = outputs["loss"]
                losses.append(accelerator.gather_for_metrics(loss.repeat(worker_batch_size)))

            losses = torch.cat(losses)
            try:
                eval_loss = torch.mean(losses)
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")

            lr_scheduler.step()
            session.report({"perplexity": perplexity, "eval_loss": eval_loss.item()}, checkpoint=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Language modeling from scratch")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--dataset-name", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--dataset-path", type=str, default="wikitext")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-iter", default=10000, type=int)
    parser.add_argument("--eval-interval", default=200, type=int)  # Iters of a epoch
    parser.add_argument("--max-sample", default=10, type=int)

    args = parser.parse_args()
    config = SEARCH_SPACE | vars(args)

    ray.init(address=None)

    trainer = ht.TorchTrainer(
        train_func,
        train_loop_config=config,
        scaling_config=ScalingConfig(num_workers=1, use_gpu=True, resources_per_worker={"CPU": 2, "GPU": 1}),
    )

    max_epoch = args.max_iter // args.eval_interval
    tuner = Tuner(
        trainer,
        param_space={"train_loop_config": config},
        tune_config=TuneConfig(
            num_samples=args.max_sample,
            metric="perplexity",
            mode="min",
        ),
        run_config=RunConfig(
            log_to_file=True,
            stop={"training_iteration": max_epoch},
            checkpoint_config=CheckpointConfig(num_to_keep=1),
        ),
    )

    results = tuner.fit()
    df = results.get_dataframe()
    df.to_csv("./ray_result.csv")
