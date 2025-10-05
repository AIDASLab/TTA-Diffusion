import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple
import transformers
from transformers.activations import ACT2FN
if transformers.__version__ == '3.4.0':
    from transformers.modeling_roberta import (
        RobertaEmbeddings,
        RobertaEncoder,
        RobertaPreTrainedModel,
        RobertaPooler,
    )
else:
    # the latest version
    from transformers.models.roberta.modeling_roberta import(
        RobertaEmbeddings,
        RobertaEncoder,
        RobertaPreTrainedModel,
        RobertaPooler,
    )

class Classifier_POS(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config, diffusion=None):
        super().__init__(config)
        self.transformer = BertModel(config)

        # Output layer that maps the hidden states to the tree label space
        self.lm_head2 = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.pos_vocab_size, bias=False)
        )

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()


    # def get_output_embeddings(self):
    #     return self.lm_head

    def set_input_embeddings(self, new_embeddings):
        self.transformer.embeddings.word_embeddings = new_embeddings

    def get_input_embeddings(self):
        return self.transformer.embeddings.word_embeddings

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
            self,
            input_ids=None,
            pos_ids=None,
            input_embs=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,           
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        labels = pos_ids
        # print(input_ids.shape, 'input_ids', )

        if inputs_embeds is None:
            # print(input_ids.shape, pos_ids.shape)
            inputs_embeds = self.transformer.embeddings.word_embeddings(input_ids)  # input_embs

        transformer_outputs = self.transformer(
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        # print(hidden_states.shape, self.lm_head2, self.lm_head2.weight.shape)
        lm_logits = self.lm_head2(hidden_states)
        # print(lm_logits.shape, hidden_states.shape, labels.shape)

        loss = None
        if pos_ids is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits
            shift_labels = labels
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


class Classifier_Tree(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = BertModel(config)

        # Output layer that maps the hidden states to the tree label space
        self.lm_head2 = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.tree_vocab_size, bias=False)
        )

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids=None,
            parse_chart=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        labels = parse_chart

        # Pass input_ids through the transformer model
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        # Construct span features for the span-based classification
        span_features = torch.cat([
            torch.unsqueeze(hidden_states, 1).expand(-1, hidden_states.size(1), -1, -1),
            torch.unsqueeze(hidden_states, 2).expand(-1, -1, hidden_states.size(1), -1)
        ], dim=-1)

        # Apply the output layer to get the logits
        lm_logits = self.lm_head2(span_features)

        loss = None
        if parse_chart is not None:
            # Compute the loss, flattening the tensor as needed for CrossEntropyLoss
            shift_logits = lm_logits
            shift_labels = parse_chart
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )

    def get_input_embeddings(self):
        return self.transformer.get_input_embeddings()
    
    ## code from air-decoding

class RobertaModel(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.pooler = RobertaPooler(config)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, token_type_ids=None, output_attentions=None, output_hidden_states=None):
        assert input_ids is not None or inputs_embeds is not None, "Either input_ids or inputs_embeds should be passed"
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        batch_size, seq_length = input_shape

        if attention_mask is None:
            attention_mask = torch.ones((batch_size,1,1, seq_length), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # print(f"input_shape: {input_shape}")
        # print(f"attention_mask_shape: {attention_mask.shape}")
        # print(f"token_type_ids_shape: {token_type_ids.shape}")

        embedding_output = self.embeddings(input_ids=input_ids,inputs_embeds=inputs_embeds, token_type_ids=token_type_ids)

        # to match the dimension: for classifier train
        # attention_mask.unsqueeze_(1).unsqueeze_(1) 

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output

class RobertaPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class RobertaLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = RobertaPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class RobertaPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = RobertaLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_ouutput, pooled_output):
        prediction_scores = self.predictions(sequence_ouutput)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score
    
class RobertaPreTrainingHeadsTopic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = RobertaLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 4)
    
    def forward(self, sequence_ouutput, pooled_output):
        prediction_scores = self.predictions(sequence_ouutput)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

class RobertaForPreTraining(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.cls = RobertaPreTrainingHeads(config)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, token_type_ids=None, sentiment=None):
        # Ensure only input_ids or inputs_embeds is passed
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")

        # Pass input embeddings or input_ids
        sequence_output, pooled_output = self.roberta(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,  # Optionally use inputs_embeds
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Get prediction scores (logits) from the classifier
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        # Return logits and optional outputs
        return seq_relationship_score  # return only the classification logits

    def get_input_embeddings(self):
        return self.roberta.embeddings.word_embeddings

class RobertaForPreTrainingTopic(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.cls = RobertaPreTrainingHeadsTopic(config)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, token_type_ids=None, sentiment=None):
        # Ensure only input_ids or inputs_embeds is passed
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")

        # Pass input embeddings or input_ids
        sequence_output, pooled_output = self.roberta(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,  # Optionally use inputs_embeds
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Get prediction scores (logits) from the classifier
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        # Return logits and optional outputs
        return seq_relationship_score  # return only the classification logits

    def get_input_embeddings(self):
        return self.roberta.embeddings.word_embeddings

class RobertaForPreTrainingSingleTopic(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.cls = RobertaPreTrainingHeads(config)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, token_type_ids=None, labels=None):
        # Ensure only input_ids or inputs_embeds is passed
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")

        # Pass input embeddings or input_ids
        sequence_output, pooled_output = self.roberta(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,  # Optionally use inputs_embeds
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Get prediction scores (logits) from the classifier
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        return SequenceClassifierOutput(
            logits=seq_relationship_score,
            hidden_states=sequence_output,
            attentions=None  # Add attentions if needed
        )

    def get_input_embeddings(self):
        return self.roberta.embeddings.word_embeddings