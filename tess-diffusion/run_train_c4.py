import logging
import os
import sys

import datasets
import nltk

import evaluate
import transformers
from transformers.trainer_callback import TrainerState
from transformers import AutoTokenizer, HfArgumentParser, set_seed, AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from sdlm.data.data_utils import load_data, tokenize_data_new
from sdlm.arguments import ModelArguments, DataTrainingArguments, DiffusionArguments, TrainingArguments
from sdlm.models import RobertaDiffusionConfig, RobertaForDiffusionLM
from sdlm.schedulers import SimplexDDPMScheduler
import pdb
from sdlm.trainer import DiffusionTrainer
from sdlm.data.data_collator import DataCollatorForSeq2Seq
from sdlm.inference.inference_utils import evaluate_generation
from sdlm.data.data_collator import SpanInfillingDataCollator
from sdlm.inference.inference_utils import process_text
from sdlm.data.postprocessors import postprocess_text_for_metric
from transformers.trainer_callback import TrainerState

from datasets import load_dataset, DatasetDict, concatenate_datasets, load_from_disk
import json

import wandb

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from sdlm.data.preprocessors import t5_random_spans_mask_batch, insert_extra_paddings, gpt_span_mask_batch
import numpy as np
import pdb
from random import choices
import torch
from enum import Enum
import random

args = {
	"model_name_or_path": 'FacebookAI/roberta-large',
	"do_train": True,
	"do_eval": True,
 	"do_predict": False,
	"dataset_name": "allenai/c4",
 	"output_dir": "${LOCAL_DIR}/outputs/paper_experiments/c4_5000",
	"per_device_train_batch_size": 64,
	"per_device_eval_batch_size": 128,
	"report_to": "wandb",
	"max_steps": 300000,
	"eval_steps": 10000,
	"max_eval_samples": 128,
 	"max_seq_length": 128,
	"conditional_generation": "prefix_lm",
    "line_by_line": True,
	"num_inference_diffusion_steps": 1000,
	"evaluation_strategy": "steps",
	"simplex_value": 5,
	"num_diffusion_steps": 5000,
	"lr_scheduler_type": "linear",
	"learning_rate": 5e-6,
	"warmup_steps": 2000,
	"weight_decay": 0.01,
	"pad_to_max_length": True,
 	"beta_schedule": "squaredcos_improved_ddpm",
	"logging_steps": 500,
	"save_steps": 10000,
    # While self-conditioning has little impact on controllability, we found that incorporating it leads to better generalization across different tasks, and thus include it as an additional configuration.
 	"self_condition": "logits_mean",
    "self_condition_mix_before_weights": True,
	"overwrite_output_dir":False,
    "autoregressive_eval_model":"gpt2-large"
}

def main():
    args_list = []
    for key, value in args.items():
        args_list.append(f"--{key}")
        args_list.append(str(value))
    
    if len(sys.argv) > 1:
        cmd_args = sys.argv[1:]  
        args_dict = {args_list[i].lstrip('-'): args_list[i+1] for i in range(0, len(args_list), 2)}
        # Parse command line args similarly
        for i in range(0, len(cmd_args), 2):
            key = cmd_args[i].lstrip('-')
            value = cmd_args[i+1]
            args_dict[key] = value  # Update or insert

        # Flatten back to args_list
        args_list = []
        for key, value in args_dict.items():
            args_list.append(f"--{key}")
            args_list.append(str(value))

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, DiffusionArguments))
    model_args, data_args, training_args, diffusion_args = parser.parse_args_into_dataclasses(args_list)

    training_args.logging_first_step=True
    training_args.log_generated_texts=False

    import logging
    logger = logging.getLogger(__name__)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # load data    

    config = RobertaDiffusionConfig.from_pretrained(
        model_args.model_name_or_path,
        self_condition=diffusion_args.self_condition,
        self_condition_zeros_after_softmax=diffusion_args.self_condition_zeros_after_softmax,
        deepmind_conditional=diffusion_args.deepmind_conditional,
        classifier_free_simplex_inputs=diffusion_args.classifier_free_simplex_inputs,
        classifier_free_uncond_input=diffusion_args.classifier_free_uncond_input,
        self_condition_mlp_projection=diffusion_args.self_condition_mlp_projection,
        self_condition_mix_before_weights=diffusion_args.self_condition_mix_before_weights,
        self_condition_mix_logits_before_weights=diffusion_args.self_condition_mix_logits_before_weights,
        empty_token_be_mask=diffusion_args.empty_token_be_mask,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if model_args.model_name_or_path:
        model = RobertaForDiffusionLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = RobertaForDiffusionLM.from_config(config)
        
    tokenized_datasets = load_from_disk("./datasets/c4_tokenized")

    ## configure datasets
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    def preprocess_logits_for_metrics(logits):
        return logits.argmax(dim=-1)

    if training_args.do_predict:
        if "test" not in tokenized_datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = tokenized_datasets["test"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            test_dataset = eval_dataset.select(range(max_eval_samples))

    ## basic perplexity eval: gpt2-large
    ## basic metric also include dist-1,2,3,4
    def get_compute_metrics(data_args, training_args, model_args):
        # Causal language model.
        causal_model = AutoModelForCausalLM.from_pretrained(model_args.autoregressive_eval_model)
        causal_model = causal_model.to(training_args.device)
        causal_tokenizer = AutoTokenizer.from_pretrained(model_args.autoregressive_eval_model)

        is_conditional_generation = data_args.conditional_generation is not None
        prefix_lm_eval = True if data_args.conditional_generation in ["prefix_lm", "ul2", "ul2_with_unconditional", "ul2_variable"] else False
        compute_metrics = lambda results: evaluate_generation(
            results,
            data_args,
            causal_model,
            causal_tokenizer,
            is_conditional_generation,
            prefix_lm_eval=prefix_lm_eval,
            skip_special_tokens=data_args.skip_special_tokens,
            eval_for_all_metrics=training_args.eval_for_all_metrics,
        )
        return compute_metrics

    vocab_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length    

    ## if data_args.conditional_generation is None, do nothing
    data_collator = lambda mode: SpanInfillingDataCollator(
        mode=mode,
        data_args=data_args,
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
        seed=training_args.seed,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        eval_context_size=10,
    )


    if training_args.do_eval:
        compute_metrics = get_compute_metrics(data_args, training_args, model_args)

    noise_scheduler = SimplexDDPMScheduler(
        num_train_timesteps=diffusion_args.num_diffusion_steps,
        beta_schedule=diffusion_args.beta_schedule,
        simplex_value=diffusion_args.simplex_value,
        clip_sample=diffusion_args.clip_sample,
        device=training_args.device,
    )
    inference_noise_scheduler = SimplexDDPMScheduler(
        num_train_timesteps=diffusion_args.num_inference_diffusion_steps,
        beta_schedule=diffusion_args.beta_schedule,
        simplex_value=diffusion_args.simplex_value,
        clip_sample=diffusion_args.clip_sample,
        device=training_args.device,
    )

    if training_args.report_to == "wandb":
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "tess_c4_4m"),
            name=args['output_dir']
        )
        wandb.config.update(args, allow_val_change=True)

    trainer = DiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not training_args.without_compute_metrics else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
        noise_scheduler=noise_scheduler,
        diffusion_args=diffusion_args,
        data_args=data_args,
        inference_noise_scheduler=inference_noise_scheduler,
    )
    
    #trainer.evaluate()

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        # TODO: num_beans should be added for ours as well.
        # metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
if __name__ == "__main__":
    main()
