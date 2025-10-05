import pdb

import numpy as np
import torch
import torch.nn.functional as F

from sdlm.metrics.metrics import distinct_n_grams, zipf
from sdlm.metrics.perplexity import conditional_perplexity, perplexity
from sdlm.metrics.repetition import repetition
from sdlm.utils import convert_to_simplex, join_texts
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu

import torch
import torch.nn.functional as F

def sample_logits(sampling_type, logits, top_p=None, temperature=1.0, top_k=1):
    """
    Sample from the logits using top-p (nucleus) sampling, top-k sampling, or greedy sampling (argmax).
    :param sampling_type: str, either "top_p", "top_k", or None for greedy sampling
    :param logits: Tensor, the logits to sample from
    :param top_p: float, the cumulative probability threshold for top-p sampling
    :param top_k: int, the number of top tokens to keep for top-k sampling
    :param temperature: float, the temperature to scale the logits
    :return: Tensor, the sampled token IDs
    """
    # Scale logits by temperature
    logits = logits / temperature
    
    if sampling_type == "top_p":
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        if top_p is not None:
            # Sort probabilities
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_keep = cumsum_probs < top_p

            # Keep the first token below the threshold
            sorted_indices_to_keep[..., 1:] = sorted_indices_to_keep[..., :-1].clone()
            sorted_indices_to_keep[..., 0] = 1

            # Scatter the sorted indices back to original positions
            indices_to_keep = sorted_indices_to_keep.scatter(dim=2, index=sorted_indices, src=sorted_indices_to_keep)
            filtered_logits = logits.masked_fill(indices_to_keep == 0, -float("Inf"))
        else:
            token_ids = torch.argmax(logits, dim=-1)
            return token_ids

    elif sampling_type == "top_k":
        if top_k is not None:
            # Find the top-k largest logits            
            top_k_logits, _ = torch.topk(logits, k=top_k, dim=-1)

            # Get the kth largest logits for each batch
            min_top_k_logits = top_k_logits[..., -1, None]

            # Filter out logits below the kth largest value
            filtered_logits = torch.where(logits < min_top_k_logits, torch.tensor(-float("Inf")).to(logits.device), logits)
        else:
            token_ids = torch.argmax(logits, dim=-1)
            return token_ids

    elif sampling_type is None:
        # Greedy sampling: choose the token with the highest probability
        token_ids = torch.argmax(logits, dim=-1)
        return token_ids

    else:
        raise NotImplementedError("Sampling type not implemented: choose 'top_p', 'top_k', or None for greedy sampling.")
    
    # Sample from the filtered distribution
    token_ids = torch.distributions.categorical.Categorical(logits=filtered_logits).sample()
    
    return token_ids



def remove_first_occurrence(string, char):
    # We do not strip as we need the spaces as well.
    if char in string:
        idx = string.index(char)
        string = string[idx + len(char) :]
    return string


def keep_till_first_occurrence(string, chars):
    """Given a list of characters, trim the text after the first occurance between them."""
    idxs = [string.index(char) for char in chars if char in string]
    if len(idxs):
        min_idx = np.min(idxs)
        string = string[:min_idx]
    return string


def process_text(texts):
    # TODO(rabeeh): for now we only cover roberta case.
    texts = [keep_till_first_occurrence(text, ["</s>"]) for text in texts]
    texts = [remove_first_occurrence(text, "<s>") for text in texts]
    return texts


def split_into_masked_and_unmasked(token_ids, span_mask, return_masked=None):
    """Given an span_mask, splits the given token_ids into masked and unmasked parts.

    If return_masked is set, only returns the masked parts, if this is set to False,
    only returns the unmasked parts, and If set to None, returns both parts.
    """

    def update_spans(span, masked, unmasked, mask):
        # TODO: this needs to be here for previous version of the codes.
        # span = torch.stack(span)
        masked.append(span) if mask else unmasked.append(span)

    masked = []
    unmasked = []
    prev_mask = span_mask[0]
    span = []
    for _, (token_id, mask) in enumerate(zip(token_ids, span_mask)):
        if mask == prev_mask:
            span.append(token_id)
        else:
            # Adds the previous span.
            update_spans(span, masked, unmasked, prev_mask)
            prev_mask = mask
            span = [token_id]
    # Adds the last span.
    update_spans(span, masked, unmasked, prev_mask)

    if return_masked is None:
        return masked, unmasked

    return masked if return_masked else unmasked


def concatenate_alternatively(longer, shorter, mark=""):
    """Given two lists of strings, concatenates them alternatively.

    We assume that the concatenated string should starts from elements in the longer
    list (which has one extra element). The shorter text can optionally be embraced with
    a `mark` text on both sides.
    """
    concatenated_str = ""
    for l, s in zip(longer, shorter):
        concatenated_str += l + " " + mark + s + mark + " "
    if len(longer) == len(shorter) + 1:
        return concatenated_str + longer[-1]
    elif len(longer) == len(shorter):
        return concatenated_str[:-1]
    else:
        raise ValueError


def aggregate_list(x):
    str = ""
    if len(x) == 0:
        return str
    for l in x:
        str += l + " "
    return str[:-1]


def logits_projection(logits, sampling_type, top_p, simplex_value, temperature, top_k):
    # TODO(rabeeh): huggingface has different sampling, like constrastive one.
    # also there are more variant in diffusion-lm.
    token_ids = sample_logits(sampling_type, logits, top_p, temperature, top_k)
    return convert_to_simplex(token_ids, simplex_value, vocab_size=logits.shape[2])


def filter_empty(texts):
    """Filters empty texts and return the remained texts and the their indices."""
    list_of_tuples = [(text, i) for i, text in enumerate(texts) if text != ""]
    if len(list_of_tuples) == 0:
        return [], []
    non_empty_texts, remained_inds = list(zip(*list_of_tuples))
    return list(non_empty_texts), list(remained_inds)


def predict_conditional_generated(span_masks, input_ids, tokenizer, predicted_token_ids, prefix_name, skip_special_tokens):
    masked = list(
        map(lambda x, y: split_into_masked_and_unmasked(x, y, return_masked=True), predicted_token_ids, span_masks)
    )
    unmasked = list(map(lambda x, y: split_into_masked_and_unmasked(x, y, return_masked=False), input_ids, span_masks))
    pred_masked_texts = [tokenizer.batch_decode(x, skip_special_tokens=skip_special_tokens) for x in masked]
    pred_unmasked_texts = [tokenizer.batch_decode(x, skip_special_tokens=skip_special_tokens) for x in unmasked]
    pred_texts = list(map(lambda x, y: concatenate_alternatively(x, y), pred_unmasked_texts, pred_masked_texts))
    pred_texts_marked = list(
        map(lambda x, y: concatenate_alternatively(x, y, mark="***"), pred_unmasked_texts, pred_masked_texts)
    )
    aggregated_masked_texts = list(map(lambda x: aggregate_list(x), pred_masked_texts))
    predicted_tokens = [np.array(item).tolist() for submasked in masked for item in submasked]
    return {
        # prefix_name: pred_texts,
        prefix_name + "_marked": pred_texts_marked,
        prefix_name + "_masked": aggregated_masked_texts,
        prefix_name + "_masked_tokens": predicted_tokens,
    }


def evaluate_generation(
    results,
    data_args,
    causal_model,
    causal_tokenizer,
    is_conditional_generation,
    prefix_lm_eval=False,
    skip_special_tokens=True,
    eval_for_all_metrics=False,
):
    metrics = {}
    # In case of prefix_lm since the generated text is unified, we can evaluate only the masked parts.
    if prefix_lm_eval:
        gold_text_key = "gold_texts_masked"
        # In case of gpt2, we only have the key of `generated_texts_masked`.
        keys = (
            ["generated_texts_masked"]
            if "generated_texts_masked" in results
            else ["pred_texts_from_simplex_masked", "pred_texts_from_logits_masked"]
        )
    else:
        keys = ["pred_texts_from_simplex", "pred_texts_from_logits"]
        gold_text_key = "gold_texts"

    if is_conditional_generation:
        gold_texts = results[gold_text_key]
        if not skip_special_tokens:
            gold_texts = process_text(gold_texts)
    if "prefixes" in results:
        prefixes = results["prefixes"]
    else:
        prefixes = None

    for key in keys:
        key_metrics = {}
        texts = results[key]
        if not skip_special_tokens:
            texts = process_text(texts)

        non_empty_texts, remained_indices = filter_empty(texts)
        if len(non_empty_texts) == 0:
            continue

        # Perplexity measured by a causal model.
        if prefixes is None:
            try: 
                key_metrics.update(
                    {"perplexity": perplexity(non_empty_texts, causal_model, causal_tokenizer)["mean_perplexity"]}
                )
            except:
                key_metrics.update(
                    {"perplexity": perplexity(non_empty_texts, causal_model, causal_tokenizer, add_start_token=False)["mean_perplexity"]}
                )
        else:
            non_empty_prefixes = [prefix for i, prefix in enumerate(prefixes) if i in remained_indices]
            perplexity_results = conditional_perplexity(non_empty_texts, non_empty_prefixes, causal_model, causal_tokenizer)
            key_metrics.update(
                {
                    "perplexity": perplexity_results["mean_perplexity"],
                    "total_perplexity": perplexity_results["mean_perplexity_total"],
                }
            )

        # Dist-1,2,3 measurements.
        key_metrics.update(distinct_n_grams(texts))
        token_dist_1, token_dist_2, token_dist_3 = eval_distinct(texts, causal_tokenizer)
        key_metrics.update({"token_dist_1": token_dist_1})
        key_metrics.update({"token_dist_2": token_dist_2})
        key_metrics.update({"token_dist_3": token_dist_3})

        # Metrics requiring the gold text.
        if is_conditional_generation and eval_for_all_metrics:
            # Note that we need to pass both context and predicted texts to this metric.
            # remained_gold_texts = [text for i, text in enumerate(gold_texts) if i in remained_indices]
            # remained_prefixes = [text for i, text in enumerate(prefixes) if i in remained_indices]
            texts_with_context = join_texts(prefixes, texts)
            gold_with_context = join_texts(prefixes, gold_texts)
            length = data_args.max_seq_length - data_args.truncation_length

        if key + "_tokens" in results and eval_for_all_metrics:
            key_metrics.update(repetition(results[key + "_tokens"], causal_tokenizer))
            key_metrics.update(zipf(results[key + "_tokens"]))

        # Adds the metrics.
        key_metrics = {f"{key}_{k}": v for k, v in key_metrics.items()}
        metrics.update(key_metrics)

    return metrics

def count_ngram(hyps_resp, n):
    """
    Count the number of unique n-grams
    :param hyps_resp: list, a list of responses
    :param n: int, n-gram
    :return: the number of unique n-grams in hyps_resp
    """
    if len(hyps_resp) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    if type(hyps_resp[0]) != list:
        print("ERROR, eval_distinct takes in a list of <class 'list'>, get a list of {} instead".format(
            type(hyps_resp[0])))
        return

    ngram = set()
    for resp in hyps_resp:
        if len(resp) < n:
            continue
        for i in range(len(resp) - n + 1):
            ngram.add(' '.join(resp[i: i + n]))
    return len(ngram)

def count_ngram_occurrences(hyps_resp, n):
    """
    Count the occurrences of each distinct n-gram.
    :param hyps_resp: list, a list of responses
    :param n: int, n-gram size
    :return: a dictionary with n-grams as keys and their counts as values
    """
    if len(hyps_resp) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    if type(hyps_resp[0]) != list:
        print("ERROR, eval_distinct takes in a list of <class 'list'>, get a list of {} instead".format(
            type(hyps_resp[0])))
        return

    ngram_counter = Counter()
    for resp in hyps_resp:
        if len(resp) < n:
            continue
        for i in range(len(resp) - n + 1):
            ngram = ' '.join(resp[i: i + n])
            ngram_counter[ngram] += 1

    return ngram_counter

def decode_and_sort_ngrams(ngram_counter, tokenizer, sort_by='count'):
    """
    Decodes the token n-grams into proper words using the tokenizer and sorts them.
    :param ngram_counter: Counter, dictionary with n-grams as keys and their counts as values
    :param tokenizer: tokenizer, the tokenizer used for decoding token IDs
    :param sort_by: str, whether to sort by 'count' (default) or 'ngram'
    :return: a sorted list of tuples (decoded n-gram, count)
    """
    decoded_ngrams = []
    
    for ngram, count in ngram_counter.items():
        token_ids = list(map(int, ngram.split()))
        decoded_ngram = tokenizer.decode(token_ids)
        decoded_ngrams.append((token_ids, decoded_ngram, count))
    
    if sort_by == 'count':
        # Sort by count in descending order
        decoded_ngrams.sort(key=lambda x: x[2], reverse=True)
    elif sort_by == 'ngram':
        # Sort by n-gram alphabetically
        decoded_ngrams.sort(key=lambda x: x[1])
    
    return decoded_ngrams


def eval_distinct(hyps_resp, tokenizer):
    """
    compute distinct score for the hyps_resp
    :param hyps_resp: list, a list of hyps responses
    :return: average distinct score for 1, 2-gram
    """

    hyps_resp = [list(map(str, tokenizer.encode(h))) for h in hyps_resp]

    if len(hyps_resp) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    if type(hyps_resp[0]) != list:
        print("ERROR, eval_distinct takes in a list of <class 'list'>, get a list of {} instead".format(
            type(hyps_resp[0])))
        return

    hyps_resp = [(' '.join(i)).split() for i in hyps_resp]
    num_tokens = sum([len(i) for i in hyps_resp])
    dist1 = count_ngram(hyps_resp, 1) / float(num_tokens)
    dist2 = count_ngram(hyps_resp, 2) / float(num_tokens)
    dist3 = count_ngram(hyps_resp, 3) / float(num_tokens)

    return dist1, dist2, dist3

def eval_repetition(hyps_resp, n):
    """
    Compute Rept-n (repetition score) for the hyps_resp
    :param hyps_resp: list of str, a list of generated responses
    :param n: int, the n-gram size to evaluate (e.g., 1, 2, 3)
    :return: Rept-n score
    """
    def count_ngram_repeats(sentences, n):
        total_ngrams = 0
        repeated_ngrams = 0
        
        for sentence in sentences:
            ngrams = [tuple(sentence[i:i + n]) for i in range(len(sentence) - n + 1)]
            total_ngrams += len(ngrams)
            repeated_ngrams += len(ngrams) - len(set(ngrams))  # Count repeated n-grams
        
        return repeated_ngrams, total_ngrams

    # Tokenize sentences into n-grams
    hyps_resp = [h.split() for h in hyps_resp]
    repeated_ngrams, total_ngrams = count_ngram_repeats(hyps_resp, n)

    # Calculate Rept-n
    if total_ngrams == 0:
        return 0.0
    return repeated_ngrams / total_ngrams

def eval_self_bleu(hyps_resp, ngram_weights=(0.25, 0.25, 0.25, 0.25)):
    """
    Compute Self-BLEU for the generated responses
    :param hyps_resp: list of str, a list of generated responses
    :param ngram_weights: tuple, n-gram weights for BLEU computation (default: BLEU-4)
    :return: Average Self-BLEU score
    """
    hyps_resp = [h.split() for h in hyps_resp]
    scores = []

    for i in range(len(hyps_resp)):
        candidate = hyps_resp[i]
        references = [ref for j, ref in enumerate(hyps_resp) if j != i]
        # Compute BLEU score for the candidate
        try:
            score = sentence_bleu(references, candidate, weights=ngram_weights)
            scores.append(score)
        except:
            print("pass candidate")
    
    return sum(scores) / len(scores) if scores else 0.0
