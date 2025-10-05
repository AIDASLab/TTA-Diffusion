"""Defines the utilities used during the training/infernece of diffusion language models."""
import torch.nn.functional as F
import os
import re
import pdb
from pathlib import Path
from transformers.utils import logging
import shutil
import numpy as np
import nltk
from typing import Callable, Iterable, List
import torch

logger = logging.get_logger(__name__)


def join_texts(prefixes, sentences):
    """Joins prefixes to setences."""
    return [f"{prefix}{sentence}" for prefix, sentence in zip(prefixes, sentences)]


def convert_to_simplex(token_ids, simplex_value, vocab_size):
    return 2 * simplex_value * F.one_hot(token_ids, vocab_size) - simplex_value


def scale(inputs, scale_value):
    return inputs / scale_value


def get_last_checkpoint(folder, prefix_checkpoint_dir="step"):
    re_checkpoint = re.compile(r"^" + prefix_checkpoint_dir + r"\_(\d+)$")
    content = os.listdir(folder)
    checkpoints = [
        path for path in content if re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(re_checkpoint.search(x).groups()[0])))


def remove_checkpoints(output_dir, checkpoint_prefix="step"):
    checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}_*") if os.path.isdir(x)]
    for checkpoint in checkpoints:
        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
        shutil.rmtree(checkpoint)


def get_norm_stats(model):
    # Gradient norm of word embeddings and lm_head.
    input_embed_grad_norm = 0
    if model.roberta.embeddings.word_embeddings.weight.grad is not None:
        input_embed_grad_norm = model.roberta.embeddings.word_embeddings.weight.grad.detach().data.norm(2).item()

    output_embed_grad_norm = 0.0
    if model.lm_head.decoder.weight.grad is not None:
        output_embed_grad_norm = model.lm_head.decoder.weight.grad.detach().data.norm(2).item()

    """
    total_grad_norm = 0.0
    for p in model.parameters():
        grad_norm = 0.0
        if  p.grad is not None:
            grad_norm = p.grad.detach().data.norm(2).item()
        total_grad_norm += grad_norm ** 2
    total_grad_norm = total_grad_norm ** 0.5

    # Norms of word embeddings and lm_head.
    input_embed_norm = model.roberta.embeddings.word_embeddings.weight.detach().data.norm(2).item()
    output_embed_norm = model.lm_head.decoder.weight.detach().data.norm(2).item()
    total_param_norm = 0.0
    for p in model.parameters():
        param_norm = p.detach().data.norm(2)
        total_param_norm += param_norm.item() ** 2
    total_param_norm = total_param_norm ** 0.5
    """
    return {
        "input_embed_grad_norm": input_embed_grad_norm,
        "output_embed_grad_norm": output_embed_grad_norm,
        # "total_grad_norm": total_grad_norm,
        # "input_embed_norm": input_embed_norm,
        # "output_embed_norm": output_embed_norm,
        # "total_param_norm": total_param_norm
    }


def self_condition_preds(self_condition, logits, logits_projection=None):
    if self_condition in ["logits", "logits_addition", "logits_mean", "logits_max", "logits_multiply"]:
        previous_pred = logits.detach()
    elif self_condition in ["logits_with_projection", "logits_with_projection_addition"]:
        previous_pred = logits_projection(logits.detach())
    else:
        assert NotImplementedError(f"{self_condition} is not implemented.")
    return previous_pred

def mix_values_based_on_self_condition(self_condition_type, value_1, value_2):
    if self_condition_type in ["logits_with_projection_addition", "logits_addition"]:
        mixed_values = value_1 + value_2
    elif self_condition_type == "logits_mean":
        mixed_values = (value_1 + value_2) / 2.0
    elif self_condition_type == "logits_max":
        mixed_values = torch.max(value_1, value_2)
    elif self_condition_type == "logits_multiply":
        mixed_values = value_1 * value_2
    else:
        assert NotImplementedError
    return mixed_values

def round_stsb_target(label):
    """STSB maps two sentences to a floating point number between 1 and 5
    representing their semantic similarity. Since we are treating all tasks as
    text-to-text tasks we need to convert this floating point number to a string.
    The vast majority of the similarity score labels in STSB are in the set
    [0, 0.2, 0.4, ..., 4.8, 5.0]. So, we first round the number to the closest
    entry in this set, and then we convert the result to a string (literally e.g.
    "3.4"). This converts STSB roughly into a 26-class classification dataset.
    Args:
      label: original label.
    Returns:
      A preprocessed label.
    """
    return np.round((label * 5) / 5, decimals=1)


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def pad_data(data_list, tokenizer):
    return tokenizer.pad({"input_ids": data_list}, padding=True)["input_ids"]


def collapse_unary_strip_pos(tree, strip_top=True):
    """Collapse unary chains and strip part of speech tags."""

    def strip_pos(tree):
        if len(tree) == 1 and isinstance(tree[0], str):
            return tree[0]
        else:
            return nltk.tree.Tree(tree.label(), [strip_pos(child) for child in tree])

    collapsed_tree = strip_pos(tree)
    collapsed_tree.collapse_unary(collapsePOS=True, joinChar="::")
    if collapsed_tree.label() in ("TOP", "ROOT", "S1", "VROOT"):
        if strip_top:
            if len(collapsed_tree) == 1:
                collapsed_tree = collapsed_tree[0]
            else:
                collapsed_tree.set_label("")
        elif len(collapsed_tree) == 1:
            collapsed_tree[0].set_label(
                collapsed_tree.label() + "::" + collapsed_tree[0].label())
            collapsed_tree = collapsed_tree[0]
    return collapsed_tree


def _get_labeled_spans(tree, spans_out, start):
    if isinstance(tree, str):
        return start + 1

    assert len(tree) > 1 or isinstance(
        tree[0], str
    ), "Must call collapse_unary_strip_pos first"
    end = start
    for child in tree:
        end = _get_labeled_spans(child, spans_out, end)
    # Spans are returned as closed intervals on both ends
    spans_out.append((start, end - 1, tree.label()))
    return end


def get_labeled_spans(tree):
    """Converts a tree into a list of labeled spans.
    Args:
        tree: an nltk.tree.Tree object
    Returns:
        A list of (span_start, span_end, span_label) tuples. The start and end
        indices indicate the first and last words of the span (a closed
        interval). Unary chains are collapsed, so e.g. a (S (VP ...)) will
        result in a single span labeled "S+VP".
    """
    tree = collapse_unary_strip_pos(tree)
    spans_out = []
    _get_labeled_spans(tree, spans_out, start=0)
    return spans_out


def chart_from_tree(label_vocab, tree):
    spans = get_labeled_spans(tree)
    num_words = len(tree.leaves())
    chart = np.full((num_words, num_words), -100, dtype=int)
    chart = np.tril(chart, -1)
    # Now all invalid entries are filled with -100, and valid entries with 0
    # print(tree)
    for start, end, label in spans:
        # Previously unseen unary chains can occur in the dev/test sets.
        # For now, we ignore them and don't mark the corresponding chart
        # entry as a constituent.
        # print(start, end, label)
        if label in label_vocab:
            chart[start, end] = label_vocab[label]
    return chart


def pad_charts(charts, padding_value=-100):
    """
    Our input text format contains START and END, but the parse charts doesn't.
    NEED TO: update the charts, so that we include these two, and set their span label to 0.

    :param charts:
    :param padding_value:
    :return:
    """
    max_len = 64
    padded_charts = torch.full(
        (len(charts), max_len, max_len),
        padding_value,
    )
    padded_charts = np.tril(padded_charts, -1)
    # print(padded_charts[-2:], padded_charts.shape)
    # print(padded_charts[1])
    for i, chart in enumerate(charts):
        # print(chart, len(chart), len(chart[0]))
        chart_size = len(chart)
        padded_charts[i, 1:chart_size + 1, 1:chart_size + 1] = chart

    # print(padded_charts[-2:], padded_charts.shape)
    return padded_charts


def remove_leaves(tree_):
    # simple_increm = 0
    for s in tree_.subtrees(lambda t: t.height() == 2):
        s[0] = '*'
        s._label = ''
    return tree_