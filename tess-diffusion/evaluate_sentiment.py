from sdlm.inference.inference_utils import process_text, evaluate_generation, process_text, eval_distinct, eval_repetition, eval_self_bleu
from sdlm.models.classifiers import RobertaForPreTraining

import os
import sys
import json
import numpy as np
import evaluate
import argparse
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

def list_of_strings(arg):
    return arg.split(',')

def normalize_text(text):
    normalized_text = text.replace('’', "'").replace('”', '"').replace('“', '"').replace('  ',' ')
    normalized_text = remove_br(normalized_text)
    normalized_text = remove_star(normalized_text)

    return normalized_text

def remove_br(text):
    text = text.replace("<br />", " ")
    return text

def remove_star(text):
    text = text.replace("***", "")
    return text

def truncate_to_n(text, n):
    tokenized_text = text.split(" ")
    if len(tokenized_text) > n:
        truncated = tokenized_text[:n]
    else:
        truncated = tokenized_text
    
    return truncated

def sentiment_accuracy(pred_sentiment, target_sentiment):
    pred_sentiment = np.array(pred_sentiment)
    target_sentiment = np.array(target_sentiment)
    #target_sentiment[target_sentiment == 2] = 1

    accuracy = np.mean(pred_sentiment == target_sentiment)

    return accuracy

def calculate_sentiment_accuracy(classifier, tokenizer, text, target, batch_size):
    target_sentiment = [target] * len(text)

    # Tokenize and batch input
    batched_input_ids = [tokenizer(text[i:i + batch_size], padding=True, return_tensors='pt')['input_ids'].to("cuda") 
                        for i in range(0, len(text), batch_size)]    
    
    # Initialize an empty list to store predictions
    pred_sentiment = []

    # Iterate through batches and get predictions
    for batch in batched_input_ids:
        predictions = classifier(batch)
        try:
            pred_sentiment_batch = np.argmax(predictions.cpu().detach().numpy(), axis=1)
        except:
            pred_sentiment_batch = np.argmax(predictions.logits.cpu().detach().numpy(), axis=1)
        pred_sentiment.extend(pred_sentiment_batch)
    
    # Compute accuracy
    accuracy = sentiment_accuracy(pred_sentiment, target_sentiment)
    return accuracy

def main():    
    parser = argparse.ArgumentParser()  
    parser.add_argument('--files', type=list_of_strings, default=None)
    parser.add_argument('--files_dir', type=str, default=None) ## if set, evaluate all json files in the directory
    parser.add_argument('--save_dir', type=str, default="./results_sentiment")  
    parser.add_argument('--perplexity_eval_model',type=str, default="gpt2-large")
    parser.add_argument('--base_acc_eval_model', type=str, default="./output/classifier_sentiment") ## classifier given in Air-decoding
    parser.add_argument('--additional_acc_eval_models', type=list_of_strings, default=["https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment"])
    parser.add_argument('--mode', type=str, default="token-diff")
    parser.add_argument('--num_prompts', type=int, default=15)
    parser.add_argument('--num_texts', type=int, default=1500)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=50) # batch size for sentiment classification
    
    args = parser.parse_args()

    files = args.files
    files_dir = args.files_dir
    save_dir = args.save_dir
    base_acc_eval_model_name = args.base_acc_eval_model
    additional_acc_eval_model_names = args.additional_acc_eval_models
    perplexity_model_name = args.perplexity_eval_model
    mode = args.mode
    num_prompts = args.num_prompts
    num_texts = args.num_texts
    num_classes =args.num_classes
    num_texts_per_prompt = int(num_texts/num_prompts)

    os.makedirs(save_dir, exist_ok=True)

    # metrics = ["dist-n", "rept-n", "perplexity", "self-bleu"]
    # eval_range = ["full_text", "per_prompt", "per_sentence"]    
    
    if files_dir is not None:
        files_list = os.listdir(files_dir)
    else:
        files_list = files

    perplexity_model = AutoModelForCausalLM.from_pretrained(perplexity_model_name, trust_remote_code=True).to('cuda')
    perplexity_tokenizer = AutoTokenizer.from_pretrained(perplexity_model_name)

    base_acc_eval_model = RobertaForPreTraining.from_pretrained(base_acc_eval_model_name).to('cuda')
    base_acc_tokenizer = AutoTokenizer.from_pretrained(base_acc_eval_model_name)
    base_acc_eval_model.eval()

    additional_acc_eval_models = [AutoModelForSequenceClassification.from_pretrained(model).to('cuda') for model in additional_acc_eval_model_names]
    additional_acc_tokenizers = [AutoTokenizer.from_pretrained(model) for model in additional_acc_eval_model_names]
    for model in additional_acc_eval_models:
        model.eval()

    for file in tqdm(files_list):
        file_name = file.split("/")[-1]
        
        # if not "json" in file_name:
        #     print(f"{file_name} passed, does not end with .json")
        #     continue

        file_name = mode + "eval_" + file_name

        if mode == "token-diff":
            if files_dir:
                file = os.path.join(files_dir, file)
            with open(file, encoding='utf-8') as f:
                target_file = json.load(f)

            full_text = target_file["text"]
            total_length = len(full_text)
            full_text = [normalize_text(text) for text in full_text]

            prompt_text_list = [[full_text[i] for i in range(idx, len(full_text), num_prompts)] for idx in range(num_prompts)]
            
            positive_text_list = full_text[int(total_length/2):]
            negative_text_list = full_text[:int(total_length/2)]
        
        
        text = {"pred_texts_from_logits": full_text,
        "pred_texts_from_simplex": full_text}
    
        print(text)

        metrics = evaluate_generation(
            text,
            None,
            perplexity_model,
            perplexity_tokenizer,
            False,
            prefix_lm_eval=False,
            skip_special_tokens=True,
            eval_for_all_metrics=True,
        )
        ## evaluate perplexity
        perplexity = metrics['pred_texts_from_logits_perplexity']

        ## mean for per sentence dist
        sentence_dist_1 = metrics['pred_texts_from_logits_dist-1']
        sentence_dist_2 = metrics['pred_texts_from_logits_dist-2']
        sentence_dist_3 = metrics['pred_texts_from_logits_dist-3']
        sentence_dist_4 = metrics['pred_texts_from_logits_dist-4']

        full_dist = eval_distinct(full_text, perplexity_tokenizer)                     

        ## mean for per prompt dist
        prompt_dist_list = [[], [], []]

        for prompt_text in prompt_text_list:
            dist_1, dist_2, dist_3 = eval_distinct(prompt_text, perplexity_tokenizer)
            prompt_dist_list[0].append(dist_1)
            prompt_dist_list[1].append(dist_2)
            prompt_dist_list[2].append(dist_3)

        # Convert to numpy array for easy manipulation
        prompt_dist_array = np.array(prompt_dist_list)

        # Compute mean for each distinct score (1, 2, 3)
        prompt_dist = np.mean(prompt_dist_array, axis=1)

        ## calaculate rept-n per sentence
        sentence_rept_1 = [eval_repetition([sentence], n=1) for sentence in full_text]
        sentence_rept_2 = [eval_repetition([sentence], n=2) for sentence in full_text]
        sentence_rept_3 = [eval_repetition([sentence], n=3) for sentence in full_text]

        sentence_rept_1 = np.mean(sentence_rept_1)
        sentence_rept_2 = np.mean(sentence_rept_2)
        sentence_rept_3 = np.mean(sentence_rept_3)

        # ## calaculate self-bleu for full text
        # self_bleu_full_text = eval_self_bleu(full_text)

        # ## calaculate self-bleu per prompt
        # self_bleu_per_prompt = [
        #     eval_self_bleu(prompt_text) for prompt_text in prompt_text_list
        # ]        
        # mean_self_bleu_per_prompt = np.mean(self_bleu_per_prompt)

        metrics_summary = {
            "perplexity": perplexity,
            "sentence_dist": {
                "dist-1": sentence_dist_1,
                "dist-2": sentence_dist_2,
                "dist-3": sentence_dist_3,
                "dist-4": sentence_dist_4,
            },
            "full_dist": {
                "dist-1": full_dist[0],
                "dist-2": full_dist[1],
                "dist-3": full_dist[2],
            },
            "prompt_dist": {
                "dist-1": prompt_dist[0],
                "dist-2": prompt_dist[1],
                "dist-3": prompt_dist[2],
            }            
        }

        # Evaluate sentiment accuracy for base model
        default_class_map = {"negative": 0, "positive": 1}    
        base_positive_acc = calculate_sentiment_accuracy(
            base_acc_eval_model, base_acc_tokenizer, positive_text_list, default_class_map['positive'], args.batch_size
        )
        base_negative_acc = calculate_sentiment_accuracy(
            base_acc_eval_model, base_acc_tokenizer, negative_text_list, default_class_map['negative'], args.batch_size
        )

        metrics_summary.update({"base_positive_acc": base_positive_acc})
        metrics_summary.update({"base_negative_acc": base_negative_acc})

        # Evaluate sentiment accuracy for additional models
        for name, classifier, tokenizer in zip(additional_acc_eval_model_names, additional_acc_eval_models, additional_acc_tokenizers):
            if name == "cardiffnlp/twitter-roberta-base-sentiment-latest":
                class_map = {"negative": 0, "positive": 2}  
                short_name = "cardiffnlp"
            elif name == "siebert/sentiment-roberta-large-english":
                class_map = {"negative": 0, "positive": 1}   
                short_name = "siebert"
            elif name == "distilbert/distilbert-base-uncased-finetuned-sst-2-english":
                class_map = {"negative": 0, "positive": 1}
                short_name = "huggingface_default"
            elif name == "j-hartmann/sentiment-roberta-large-english-3-classes":
                class_map = {"negative": 0, "positive": 2}
                short_name = "hartmann"
            
            # Positive and negative sentiment accuracy
            positive_acc = calculate_sentiment_accuracy(
                classifier, tokenizer, positive_text_list, class_map['positive'], args.batch_size
            )
            negative_acc = calculate_sentiment_accuracy(
                classifier, tokenizer, negative_text_list, class_map['negative'], args.batch_size
            )

            # Update metrics
            metrics_summary.update({short_name + "_positive_acc": positive_acc})
            metrics_summary.update({short_name + "_negative_acc": negative_acc})
        
        print(metrics_summary)
        acc_list = []
        for key, value in metrics_summary.items():
            if "acc" in key:
                acc_list.append(value)
        acc_list = np.array(acc_list)
        acc_mean = np.mean(acc_list)
        print(f"Mean accuracy: {acc_mean}")

        save_path = os.path.join(save_dir, file_name)
        with open(save_path, 'w') as json_file:
            json.dump(metrics_summary, json_file, indent=4)

        print(f"Metrics saved to: {save_path}")

if __name__=="__main__":
    main()
        