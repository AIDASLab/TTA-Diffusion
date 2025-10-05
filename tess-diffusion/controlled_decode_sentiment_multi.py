import logging
import os
import sys

import datasets
from datasets import load_from_disk, Dataset, DatasetDict, load_dataset
import nltk
import json
import numpy as np
import torch
import transformers
import time

import evaluate
import transformers
from transformers import pipeline
from transformers.trainer_callback import TrainerState
from transformers import AutoTokenizer, HfArgumentParser, set_seed, AutoModelForTokenClassification, AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from sdlm.data.data_utils import load_data
from sdlm.arguments import ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, DiffusionArguments, TrainingArguments, DecodingArguments
from sdlm.models import RobertaDiffusionConfig, RobertaForDiffusionLM, BertForDiffusionLM, BertDiffusionConfig
from sdlm.schedulers import SimplexDDPMScheduler, ModifiedSimplexDDPMScheduler
from sdlm.trainer import DiffusionTrainerForControl
from sdlm.control_utils import pad_charts, chart_from_tree
import pdb
#from sdlm.trainer import DiffusionTrainer
from sdlm.data.data_collator import DataCollatorForSeq2Seq, SpanInfillingDataCollator, CustomDataCollatorForSentiment
from sdlm.inference.inference_utils import process_text, evaluate_generation, process_text, eval_distinct
from sdlm.data.postprocessors import postprocess_text_for_metric
from sdlm.models.classifiers import RobertaForPreTraining

from dataclasses import dataclass, field
from typing import Optional
import random
import gc
import subprocess

def parse_command_line_args():
    # This basic parser will only handle command-line inputs that overlap with the keys in `args`
    import argparse
    parser = argparse.ArgumentParser()

    for key in args:
        parser.add_argument(f"--{key}", type=type(args[key]), default=args[key])

    # Parse known args from the command line
    parsed_args, _ = parser.parse_known_args()
    return vars(parsed_args)

args = {
	"model_name_or_path": "roberta-large",
 	"do_predict": True,
	"dataset_name": "tuetschek/e2e_nlg",
 	"output_dir": "./results/imdb_sentiment_c4_penalty_renew",
	"per_device_train_batch_size": 50,
	"per_device_eval_batch_size": 200,
	"report_to": "tensorboard",
	"max_steps": 20000,
	"max_steps": 20000,
	"eval_steps": 5000,
	#"max_eval_samples": 128,
 	"max_seq_length": 64,
	"line_by_line": True,
	"conditional_generation": "seq2seq",
	#"num_inference_diffusion_steps": 1000,
    "num_inference_diffusion_steps": 1000,
	"evaluation_strategy": "steps",
	"simplex_value": 5,
    #"simplex_value": 10,
	"num_diffusion_steps": 2000,
	"lr_scheduler_type": "linear",
	"learning_rate": 1e-4,
	"warmup_steps": 2000,
	"weight_decay": 0.01,
	"pad_to_max_length": True,
 	"beta_schedule": "squaredcos_improved_ddpm",
	"logging_steps": 1000,
	"save_steps": 1000,
 	"self_condition": "logits_mean",
    "self_condition_mix_before_weights": True,
    "classifier_control": "sentiment",
    "pos_classifier": "./output/classifier_pos_token/checkpoint-1800",
	"tree_classifier": "./output/classifier_tree/checkpoint-7500",
    #"sentiment_classifier": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "sentiment_classifier": None,
    "sub_classifiers": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "control_lambda": 2000,
    "control_iteration": 3,
    "eval_per_n_step": 200,    
    "num_generate_samples": 50,
	"perplexity_model_name_or_path": "gpt2-large",
    "lambda_schedule": "constant",
    "lambda_alpha": 1.0,
    "lambda_beta": 0.0,
	"token_schedule": "constant",
    "token_alpha": 1.0,
    "token_beta": 0.0,
    "checkpoint_dir": "./${LOCAL_DIR}/outputs/paper_experiments/c4_additional",
    #"checkpoint_dir":"/home/work/woojin/projects/tess-diffusion/${LOCAL_DIR}/outputs/paper_experiments/openwebtext_roberta",
    "checkpoint_file": "checkpoint-300000",
    "sampling_type": "top_p",
    "top_p": 0.9,
    "temperature": 1.0,
    "scale_initial_noise": 1.0,
    "scale_intermediate_noise": 1.0,
    "classifier_begin_step": 1000,
    "classifier_end_step": 0,
    "classifier_interval": 1,
    "initial_noise_type": "randn",
    "intermediate_noise_type": "randn",
    "ddim": False,
    "schedule_warm_up_step": 0,
    "fluency_classifier_name_or_path": "textattack/roberta-base-CoLA",
    "fluency_classifier_begin_step": 1000,
    "fluency_classifier_end_step": 0,
    "fluency_classifier_interval": 1,
    "fluency_classifier_iteration": 1,
    "fluency_classifier_lambda": 2000,
    "monotonic": False,
    "ignore_prompts": False,
}

def main():
    # Parse command-line arguments
    cmd_args = parse_command_line_args()

    # Update the predefined args with any command-line arguments
    args.update(cmd_args)

    # Convert the updated args dictionary into a list of arguments
    args_list = []
    for key, value in args.items():        
        args_list.append(f"--{key}")
        args_list.append(str(value))    

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, DiffusionArguments, DecodingArguments))
    model_args, data_args, training_args, diffusion_args, decoding_args = parser.parse_args_into_dataclasses(args_list)    
    
    decoding_args.classifier_model_name_or_path = None

    if decoding_args.sub_classifiers is not None:
        decoding_args.sub_classifiers = decoding_args.sub_classifiers[0].split(",")
    
    classifier_name = ""

    default_class_map = {"negative": 0, "positive": 1}
    default_class_inverse_map = {v: k for (k, v) in default_class_map.items()}
    class_map = {"negative": 0, "positive": 1}
    inverse_class_map = {v: k for (k, v) in default_class_map.items()}

    if "cardiffnlp/twitter-roberta-large-topic-sentiment-latest" in decoding_args.sub_classifiers:        
        classifier_name += "_twitter_large"
    if "cardiffnlp/twitter-roberta-base-sentiment-latest" in decoding_args.sub_classifiers:        
        classifier_name += "_twitter_base"
    if "Cloudy1225/stackoverflow-roberta-base-sentiment" in decoding_args.sub_classifiers:
        classifier_name += "_stack"
    if "siebert/sentiment-roberta-large-english" in decoding_args.sub_classifiers: 
        classifier_name += "_siebert"
    if "j-hartmann/sentiment-roberta-large-english-3-classes" in decoding_args.sub_classifiers: 
        classifier_name += "_hartmann"
    if "./output/classifier_sentiment_new" in decoding_args.sub_classifiers:       
        classifier_name += "_default"
    if "textattack/roberta-base-CoLA" in decoding_args.sub_classifiers:
        classifier_name += "_cola"
    
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

    prompts = load_dataset("json", data_files="datasets/imdb/prompt_sent.json")    
    
    def tokenize_data(example):
        text = example['prompt']
        tokenized_input_ids = [tokenizer.bos_token_id]
        tokenized_prompt = tokenizer(text, add_special_tokens=False)

        tokenized_input_ids.extend(tokenized_prompt['input_ids'])

        example['input_ids'] = tokenized_input_ids
        return example

    def duplicate_rows_add_sent(example):
        # Repeat each row x 2 (num_sents) x num_samples
        num_samples = decoding_args.num_generate_samples
        num_prompts = 15       
        num_sents = 2
        return {
            "prompt": [prompt for prompt in example["prompt"]] * num_sents * num_samples,
            "input_ids": [ids for ids in example["input_ids"]] * num_sents * num_samples,
            "sentiment": [class_map['negative']]*num_prompts * num_samples + [class_map['positive']]* num_prompts * num_samples            
        }
    
    tokenized_prompts = prompts.map(tokenize_data)
    #tokenized_prompts['train'] = tokenized_prompts['train'].select(range(1))
    print(tokenized_prompts['train'])

    tokenized_eval_dataset = tokenized_prompts['train'].map(duplicate_rows_add_sent, batched=True)
    tokenized_eval_dataset = tokenized_eval_dataset.flatten_indices()
    print(tokenized_eval_dataset['sentiment'])

    # Define the data collator function
    data_collator = lambda mode: CustomDataCollatorForSentiment(
        tokenizer,
        padding="max_length" if data_args.pad_to_max_length else True,
        max_length=data_args.max_seq_length,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    noise_scheduler = SimplexDDPMScheduler(
        num_train_timesteps=diffusion_args.num_diffusion_steps,
        beta_schedule=diffusion_args.beta_schedule,
        simplex_value=diffusion_args.simplex_value,
        clip_sample=diffusion_args.clip_sample,
        device=training_args.device,
    )
    inference_noise_scheduler = ModifiedSimplexDDPMScheduler(
        num_train_timesteps=diffusion_args.num_inference_diffusion_steps,
        beta_schedule=diffusion_args.beta_schedule,
        simplex_value=diffusion_args.simplex_value,
        clip_sample=diffusion_args.clip_sample,
        device=training_args.device,
        token_schedule=decoding_args.token_schedule,
        token_alpha=decoding_args.token_alpha,
        token_beta=decoding_args.token_beta,
        ddim = decoding_args.ddim,
        schedule_warm_up_step = decoding_args.schedule_warm_up_step,
        monotonic = decoding_args.monotonic,
        ignore_prompts= decoding_args.ignore_prompts
    )

    def preprocess_logits_for_metrics(logits):
        return logits.argmax(dim=-1)

    # Initialize our Trainer
    trainer = DiffusionTrainerForControl(
        model=model,
        args=training_args,
        # train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=tokenized_eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        #compute_metrics=compute_metrics if (training_args.do_eval or training_args.do_predict) else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if (training_args.do_eval or training_args.do_predict) else None,
        noise_scheduler=noise_scheduler,
        diffusion_args=diffusion_args,
        data_args=data_args,
        decoding_args=decoding_args,
        inference_noise_scheduler=inference_noise_scheduler,
    )

    checkpoint_dir = decoding_args.checkpoint_dir
    checkpoint_file = decoding_args.checkpoint_file
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    trainer._load_from_checkpoint(checkpoint_path)
    print(f"{checkpoint_file} loaded")
    
    trainer.args = training_args
    trainer.data_args = data_args
    trainer.diffusion_args = diffusion_args
    trainer.decoding_args = decoding_args
    
    print(f"eval_per_n_step: {trainer.decoding_args.eval_per_n_step}")
    print(f"eval_batch_size: {trainer.args.per_device_eval_batch_size}")
    print(f"lambda: {trainer.decoding_args.control_lambda}")
    print(f"token_schedule: {trainer.decoding_args.token_schedule}")

    #test_dataset = tokenized_eval_dataset.select(random.sample(range(1500), 100))
    test_dataset = tokenized_eval_dataset

    test_dataloader = trainer.get_eval_dataloader(test_dataset)

    start_time = time.time()
    output = trainer.evaluation_loop(test_dataloader, 'evaluation', metric_key_prefix="test")
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"elapsed_time: {elapsed_time}")

    # causal_model = AutoModelForCausalLM.from_pretrained(decoding_args.perplexity_model_name_or_path)
    # causal_model = causal_model.to('cuda')
    # causal_tokenizer = AutoTokenizer.from_pretrained(decoding_args.perplexity_model_name_or_path)

    # def eval_perplexity(text):
    #     perplexity = evaluate_generation(
    #             text,
    #             args,
    #             causal_model,
    #             causal_tokenizer,
    #             False,
    #             prefix_lm_eval=False,
    #             skip_special_tokens=True,
    #             eval_for_all_metrics=True,
    #         )

    #     return perplexity    
    
    # def remove_br(text):
    #     text = text.replace("<br />", " ")
    #     return text

    # def remove_star(text):
    #     text = text.replace("***", "")
    #     return text

    # if decoding_args.classifier_control == "sentiment":
    #     sentiment_classifier = RobertaForPreTraining.from_pretrained("./output/classifier_sentiment")
    #     sentiment_classifier = sentiment_classifier.to('cuda')

    #     sentiment_classifier.eval()

    #     def sentiment_accuracy(pred_sentiment, target_sentiment):
    #         pred_sentiment = np.array(pred_sentiment)
    #         target_sentiment = np.array(target_sentiment)
    #         #target_sentiment[target_sentiment == 2] = 1

    #         accuracy = np.mean(pred_sentiment == target_sentiment)

    #         return accuracy

    #     text = {"pred_texts_from_logits": output.results['pred_texts_from_logits_marked'],
    #         "pred_texts_from_simplex": output.results['pred_texts_from_simplex_marked']}

    #     cleaned_text_lst = [remove_br(remove_star(text)) for text in text['pred_texts_from_logits']]
    #     text["pred_texts_from_logits"] = cleaned_text_lst
    #     text["pred_texts_from_simplex"] = cleaned_text_lst

    #     target_sentiment_lst = test_dataset['sentiment']
    #     target_sentiment_lst = [default_class_map[inverse_class_map[tgt]] for tgt in target_sentiment_lst]
    #     print(target_sentiment_lst)

    #     batch_size = 100  # You can adjust the batch size depending on your GPU memory

    #     # Split the input into batches
    #     batched_input_ids = [tokenizer(cleaned_text_lst[i:i + batch_size], padding=True, return_tensors='pt')['input_ids'].to("cuda") 
    #                         for i in range(0, len(cleaned_text_lst), batch_size)]

    #     # Initialize an empty list to store the predictions
    #     pred_sentiment_lst = []

    #     # Iterate through the batches and process them
    #     for batch in batched_input_ids:
    #         pred_sentiment = sentiment_classifier(batch)
    #         pred_sentiment_lst_batch = np.argmax(pred_sentiment.cpu().detach().numpy(), axis=1)
    #         pred_sentiment_lst.extend(pred_sentiment_lst_batch)

    #     accuracy = sentiment_accuracy(pred_sentiment_lst, target_sentiment_lst)

    #     perplexity = eval_perplexity(text)['pred_texts_from_logits_perplexity']
    #     dist_metric = eval_distinct(cleaned_text_lst, tokenizer)

    #     metric = {"task": decoding_args.classifier_control,
    #             "control_lambda:": decoding_args.control_lambda,
    #             "dist_metric": dist_metric,
    #             "perplexity": perplexity,
    #             "sentiment_accuracy": accuracy,
    #             }
    #     print(metric)
        
    #     metric.update({"text": cleaned_text_lst})   

    #     file_name = f"classifier_{classifier_name}_ddim_{decoding_args.ddim}_samples_{decoding_args.num_generate_samples}_topp_{diffusion_args.top_p}_infer_steps_{diffusion_args.num_inference_diffusion_steps}_temp_{diffusion_args.temperature}_simplex_{diffusion_args.simplex_value}_cse_{decoding_args.classifier_begin_step}_{decoding_args.classifier_end_step}_n_type_{decoding_args.initial_noise_type}_{decoding_args.intermediate_noise_type}_n_scale_{decoding_args.scale_initial_noise}_{decoding_args.scale_intermediate_noise}_iteration_{decoding_args.control_iteration}_lambda_{decoding_args.control_lambda}_schedule_{decoding_args.lambda_schedule}__{decoding_args.lambda_alpha}_{decoding_args.lambda_beta}_token_{decoding_args.token_schedule}_{decoding_args.token_alpha}_{decoding_args.token_beta}.json"
    #     save_path = os.path.join(training_args.output_dir, file_name)
        
    #     print(f"saving to {file_name}")

    #     with open(save_path, "w") as f:
    #         json.dump(metric, f)
    # Clean up functions
    def remove_br(text):
        return text.replace("<br />", " ")

    def remove_star(text):
        return text.replace("***", "")

    # Only save text part, skip evaluation
    if decoding_args.classifier_control == "sentiment":
        text = {
            "pred_texts_from_logits": output.results['pred_texts_from_logits_marked'],
            "pred_texts_from_simplex": output.results['pred_texts_from_simplex_marked']
        }

        # Clean both types
        cleaned_text_lst = [remove_br(remove_star(t)) for t in text['pred_texts_from_logits']]
        text["pred_texts_from_logits"] = cleaned_text_lst
        text["pred_texts_from_simplex"] = cleaned_text_lst  # Reuse for compatibility

        # Prepare text-only JSON
        output_json = {
            "text": cleaned_text_lst
        }

        # Save to output
        file_name = f"{checkpoint_file}_classifier_{classifier_name}_ddim_{decoding_args.ddim}_samples_{decoding_args.num_generate_samples}_topp_{diffusion_args.top_p}_infer_steps_{diffusion_args.num_inference_diffusion_steps}_warm_{decoding_args.schedule_warm_up_step}_scond_{diffusion_args.self_condition}_monotonic_{decoding_args.monotonic}_igprom_{decoding_args.ignore_prompts}_cse_{decoding_args.classifier_begin_step}_{decoding_args.classifier_end_step}_iteration_{decoding_args.control_iteration}_lambda_{decoding_args.control_lambda}_schedule_{decoding_args.lambda_schedule}__{decoding_args.lambda_alpha}_{decoding_args.lambda_beta}_token_{decoding_args.token_schedule}_{decoding_args.token_alpha}_{decoding_args.token_beta}.json"

        save_path = os.path.join(training_args.output_dir, file_name)
        print(f"Saving generated text only to {file_name}")

        with open(save_path, "w") as f:
            json.dump(output_json, f)

        torch.cuda.empty_cache()
        gc.collect()
        
        # Call evaluation script
        eval_script = "evaluate_sentiment.py"
        eval_command = [
            "python", eval_script,
            "--files", save_path,
            "--save_dir", os.path.join(training_args.output_dir, "eval"),
            "--additional_acc_eval_models",
            "siebert/sentiment-roberta-large-english,j-hartmann/sentiment-roberta-large-english-3-classes",
            "--mode", "token-diff"
        ]

        print(f"Running evaluation: {' '.join(eval_command)}")
        subprocess.run(eval_command)

if __name__ == "__main__":
    main()
    
    gc.collect()
    torch.cuda.empty_cache()