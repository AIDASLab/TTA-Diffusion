from typing import Optional, Tuple, Union

import torch
import pdb
from diffusers.pipeline_utils import DiffusionPipeline
from sdlm.inference.inference_utils import logits_projection
from sdlm.utils import scale, self_condition_preds
from dataclasses import dataclass
import numpy as np
from diffusers.utils import BaseOutput
from sdlm.utils import convert_to_simplex
import torch.nn.functional as F
from sklearn.metrics import f1_score
import torch.optim as optim
import random


@dataclass
class SimplexDiffusionPipelineOutput(BaseOutput):
    """
    Output class for simplex diffusion pipelines.
    Args:
        simplex (`np.ndarray`)
            numpy array showing the denoised simplex representation.
        logits (`np.ndarray`) final generated logits before applying the projection.
    """

    simplex: np.ndarray
    logits: np.ndarray
    loss: np.ndarray


class SimplexDDPMPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Parameters:
        model: Model architecture to denoise the latents (encoded token ids).
        scheduler ([`SchedulerMixin`]): A scheduler to denoise the encoded latent.
    """

    def __init__(
        self,
        model,
        scheduler,
        simplex_value,
        top_p,
        sampling_type,
        is_conditional_generation,
        tokenizer,
        classifier_free_uncond_input,
        temperature,
        guidance_softmax_combination
    ):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)
        self.simplex_value = simplex_value
        self.top_p = top_p
        self.sampling_type = sampling_type
        self.is_conditional_generation = is_conditional_generation
        self.tokenizer = tokenizer
        self.classifier_free_uncond_input = classifier_free_uncond_input
        self.temperature = temperature
        self.guidance_softmax_combination=guidance_softmax_combination

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        seq_length: int = 512,
        generator: Optional[torch.Generator] = None,
        batch: Optional[torch.FloatTensor] = None,
        guidance_scale: float = 1.0,
    ) -> Union[SimplexDiffusionPipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            seq_length: (`int`), sequence length for the generated samples.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            batch (`torch.FloatTensor`): batch of input data, mostly used in the conditional generation setting.
        Returns:
            [`~pipeline_utils.SimplexDiffusionPipelineOutput`]: returns the generated simplex.
        """
        # Classifier_free guidance works only in the conditional generation case.
        classifier_free_guidance = guidance_scale > 1.0 and self.is_conditional_generation
        """
        if classifier_free_guidance:
            # Makes unconditional input for max sequence length, later we truncate it.
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=seq_length, return_tensors="pt"
            ).to(self.device)
            # Converts this to a simplex (batch_size, max_seq, vocab_size)
            uncond_simplex = convert_to_simplex(uncond_input["input_ids"], self.simplex_value, self.model.config.vocab_size)
        """
        # Sample gaussian noise to begin loop
        vocab_size = self.model.config.vocab_size
        if batch is not None:
            # TODO(rabeeh): is giving the length cheating for this setting?
            # Adapts the sequence length to the given `span_mask`'s length.
            seq_length = batch["input_ids"].shape[1]
        simplex_shape = (batch_size, seq_length, vocab_size)
        simplex = self.simplex_value * torch.randn(simplex_shape, generator=generator, device=self.device)
        if self.model.config.self_condition is not None:
            previous_pred = torch.zeros((batch_size, seq_length, vocab_size), device=self.device)
        logits_projection_fct = lambda x: logits_projection(
            x, self.sampling_type, self.top_p, self.simplex_value, self.temperature, None
        )

        for t in self.progress_bar(self.scheduler.timesteps):
            # TODO(rabeeh): also check without the scale.
            t_scaled = scale(t, len(self.scheduler))
            """
            if classifier_free_guidance:
                if self.classifier_free_uncond_input == "empty_token":
                    uncond_input = uncond_simplex[:, : batch["input_ids"].shape[1], :]
                elif self.classifier_free_uncond_input == "noisy_simplex":
                    uncond_input = self.simplex_value * torch.randn(simplex.shape, generator=generator, device=self.device)
                else:
                    raise NotImplementedError
            """
            # 1. predict noise model_output. Note we need not to pass the input_ids in case of
            # unconditional generation since the loss would be computed and it should not.
            model_output = self.model(
                input_ids=batch["input_ids"] if self.is_conditional_generation else None,
                span_mask=batch["span_mask"] if self.is_conditional_generation else None,
                simplex=simplex,
                timesteps=t_scaled,
                previous_pred=previous_pred if self.model.config.self_condition else None,
                classifier_free_guidance=classifier_free_guidance,
                # unconditional_simplex=uncond_input if classifier_free_guidance else None,
            )
            model_output_logits = model_output.logits

            # Performs classifier-free guidance.
            if classifier_free_guidance:
                logits_uncond, logits_pred = model_output_logits.chunk(2)
                if self.guidance_softmax_combination:
                    model_output_logits = F.softmax(logits_uncond, dim=-1) +  guidance_scale * (F.softmax(logits_pred, dim=-1) - F.softmax(logits_uncond, dim=-1))
                else:
                    model_output_logits = logits_uncond + guidance_scale * (logits_pred - logits_uncond)

            if self.model.config.self_condition is not None:
                if classifier_free_guidance:
                    prev_output_logits = model_output.logits.chunk(2)[1]
                else:
                    prev_output_logits = model_output_logits

                previous_pred = self_condition_preds(
                    self.model.config.self_condition, prev_output_logits, logits_projection_fct
                )

            # Projection.
            projected_logits = logits_projection_fct(model_output_logits)

            # 2. compute previous logits: x_t -> x_t-1
            noise = self.simplex_value * torch.randn(simplex_shape, generator=generator, device=self.device)
            simplex = self.scheduler.step(projected_logits, t, noise, generator=generator).prev_sample

        return SimplexDiffusionPipelineOutput(simplex=simplex, logits=model_output_logits, loss=model_output.loss)

   
class ModifiedSimplexDDPMPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Parameters:
        model: Model architecture to denoise the latents (encoded token ids).
        scheduler ([`SchedulerMixin`]): A scheduler to denoise the encoded latent.
    """

    def __init__(
        self,
        model,
        scheduler,
        simplex_value,
        top_p,
        top_k,
        sampling_type,
        is_conditional_generation,
        tokenizer,
        classifier_free_uncond_input,
        temperature,
        guidance_softmax_combination,
        classifier_control,
        classifier_model,
        control_lambda,
        control_iteration,
        eval_per_n_step,
        lambda_schedule,
        lambda_alpha,
        lambda_beta,
        sub_classifier,
        sub_classifiers,
        scale_initial_noise,
        scale_intermediate_noise,
        classifier_begin_step,
        classifier_end_step,
        classifier_interval,
        initial_noise_type,
        intermediate_noise_type,
        token_schedule,
        schedule_warm_up_step,
        fluency_classifier,
        fluency_classifier_begin_step,
        fluency_classifier_end_step,
        fluency_classifier_interval,
        fluency_classifier_lambda,
        fluency_classifier_iteration
    ):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)
        self.simplex_value = simplex_value
        self.top_p = top_p
        self.top_k = top_k
        self.sampling_type = sampling_type
        self.is_conditional_generation = is_conditional_generation
        self.tokenizer = tokenizer
        self.classifier_free_uncond_input = classifier_free_uncond_input
        self.temperature = temperature
        self.guidance_softmax_combination=guidance_softmax_combination
        self.classifier_control = classifier_control
        self.classifier_model = classifier_model
        self.control_lambda = control_lambda
        self.control_iteration = control_iteration
        self.eval_per_n_step = eval_per_n_step
        self.lambda_schedule = lambda_schedule
        self.lambda_alpha = lambda_alpha
        self.lambda_beta = lambda_beta
        self.sub_classifier = sub_classifier
        self.sub_classifiers = sub_classifiers
        self.scale_initial_noise = scale_initial_noise
        self.scale_intermediate_noise = scale_intermediate_noise
        self.classifier_begin_step = classifier_begin_step
        self.classifier_end_step = classifier_end_step
        self.initial_noise_type = initial_noise_type
        self.intermediate_noise_type = intermediate_noise_type
        self.token_schedule = token_schedule
        self.token_grad = None
        self.max_timesteps = self.scheduler.num_train_timesteps
        self.classifier_interval = classifier_interval
        self.schedule_warm_up_step = schedule_warm_up_step
        self.fluency_classifier = fluency_classifier
        self.fluency_classifier_begin_step = fluency_classifier_begin_step
        self.fluency_classifier_end_step = fluency_classifier_end_step
        self.fluency_classifier_interval = fluency_classifier_interval
        self.fluency_classifier_lambda = fluency_classifier_lambda
        self.fluency_classifier_iteration = fluency_classifier_iteration
        

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        seq_length: int = 512,
        generator: Optional[torch.Generator] = None,
        batch: Optional[torch.FloatTensor] = None,
        guidance_scale: float = 1.0,
    ) -> Union[SimplexDiffusionPipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            seq_length: (`int`), sequence length for the generated samples.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            batch (`torch.FloatTensor`): batch of input data, mostly used in the conditional generation setting.
        Returns:
            [`~pipeline_utils.SimplexDiffusionPipelineOutput`]: returns the generated simplex.
        """
        # Classifier_free guidance works only in the conditional generation case.
        classifier_free_guidance = guidance_scale > 1.0 and self.is_conditional_generation
        """
        if classifier_free_guidance:
            # Makes unconditional input for max sequence length, later we truncate it.
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=seq_length, return_tensors="pt"
            ).to(self.device)
            # Converts this to a simplex (batch_size, max_seq, vocab_size)
            uncond_simplex = convert_to_simplex(uncond_input["input_ids"], self.simplex_value, self.model.config.vocab_size)
        """
        # Sample gaussian noise to begin loop
        vocab_size = self.model.config.vocab_size
        if batch is not None:
            # TODO(rabeeh): is giving the length cheating for this setting?
            # Adapts the sequence length to the given `span_mask`'s length.
            seq_length = batch["input_ids"].shape[1]
        simplex_shape = (batch_size, seq_length, vocab_size)
        random_tensor = self.generate_random_tensor(self.initial_noise_type, simplex_shape, device=self.device, generator=generator)
        #simplex = self.simplex_value * torch.randn(simplex_shape, generator=generator, device=self.device) * self.scale_initial_noise
        simplex = self.simplex_value * random_tensor * self.scale_initial_noise
        if self.model.config.self_condition is not None:
            previous_pred = torch.zeros((batch_size, seq_length, vocab_size), device=self.device)
        logits_projection_fct = lambda x: logits_projection(
            x, self.sampling_type, self.top_p, self.simplex_value, self.temperature, self.top_k
        )

        count = 0

        for t in self.progress_bar(self.scheduler.timesteps):
            # TODO(rabeeh): also check without the scale.
            t_scaled = scale(t, len(self.scheduler))
            """
            if classifier_free_guidance:
                if self.classifier_free_uncond_input == "empty_token":
                    uncond_input = uncond_simplex[:, : batch["input_ids"].shape[1], :]
                elif self.classifier_free_uncond_input == "noisy_simplex":
                    uncond_input = self.simplex_value * torch.randn(simplex.shape, generator=generator, device=self.device)
                else:
                    raise NotImplementedError
            """
            # 1. predict noise model_output. Note we need not to pass the input_ids in case of
            # unconditional generation since the loss would be computed and it should not.
            model_output = self.model(
                input_ids=batch["input_ids"] if self.is_conditional_generation else None,
                span_mask=batch["span_mask"] if self.is_conditional_generation else None,
                simplex=simplex,
                timesteps=t_scaled,
                previous_pred=previous_pred if self.model.config.self_condition else None,
                classifier_free_guidance=classifier_free_guidance,
                # unconditional_simplex=uncond_input if classifier_free_guidance else None,
            )
            model_output_logits = model_output.logits

            # Performs classifier-free guidance.
            if classifier_free_guidance:
                logits_uncond, logits_pred = model_output_logits.chunk(2)
                if self.guidance_softmax_combination:
                    model_output_logits = F.softmax(logits_uncond, dim=-1) +  guidance_scale * (F.softmax(logits_pred, dim=-1) - F.softmax(logits_uncond, dim=-1))
                else:
                    model_output_logits = logits_uncond + guidance_scale * (logits_pred - logits_uncond)

            if self.model.config.self_condition is not None:
                if classifier_free_guidance:
                    prev_output_logits = model_output.logits.chunk(2)[1]
                else:
                    prev_output_logits = model_output_logits

                previous_pred = self_condition_preds(
                    self.model.config.self_condition, prev_output_logits, logits_projection_fct
                )
            # # # adjust logits based on control
            if self.classifier_control == "pos":
                model_output_logits = self.apply_pos_control(t, model_output_logits, batch)

            elif self.classifier_control == "tree":
                model_output_logits = self.apply_tree_control(t, model_output_logits, batch)

            elif self.classifier_control == "span":
                model_output_logits = self.apply_span_control(t, model_output_logits, batch)

            elif self.classifier_control == "length":
                model_output_logits = self.apply_length_control(t, model_output_logits, batch)
                
            elif self.classifier_control == "sentiment":
                model_output_logits = self.apply_sentiment_control(t, model_output_logits, batch)

            elif self.classifier_control == "topic":
                model_output_logits = self.apply_topic_control(t, model_output_logits, batch)
                
            elif self.classifier_control == "toxicity":
                model_output_logits = self.apply_toxicity_control(t, model_output_logits, batch)
            
                           
            else:
                model_output_logits = model_output_logits

            # Projection.
            projected_logits = logits_projection_fct(model_output_logits)

            # 2. compute previous logits: x_t -> x_t-1
            random_tensor = self.generate_random_tensor(self.intermediate_noise_type, simplex_shape, device=self.device, generator=generator)            
            
            noise = self.simplex_value * random_tensor * self.scale_intermediate_noise

            prompt_mask = batch["span_mask"] == 0 if self.scheduler.ignore_prompts else None

            simplex = self.scheduler.step(projected_logits, t, noise, generator=generator, token_grad=self.token_grad, prompt_mask=prompt_mask,).prev_sample

        return SimplexDiffusionPipelineOutput(simplex=simplex, logits=model_output_logits, loss=model_output.loss)
    
    def generate_random_tensor(self, random_type, simplex_shape, device, generator=None, input_1=None, input_2=None):   
        if random_type == "randn":  # Normal distribution
            mean = input_1 if input_1 is not None else 0.0
            variance = input_2 if input_2 is not None else 1.0
            random_tensor = torch.randn(simplex_shape, generator=generator, device=device) * variance + mean
        elif random_type == "ones":
            random_tensor = torch.ones(simplex_shape, device=device)
        elif random_type == "zeros":
            random_tensor = torch.zeros(simplex_shape, device=device)
        elif random_type == "uniform":  # Uniform distribution
            low = input_1 if input_1 is not None else 0.0
            high = input_2 if input_2 is not None else 1.0
            random_tensor = torch.empty(simplex_shape, device=device).uniform_(low, high)
        elif random_type == "constant":
            constant_value = input_1 if input_1 is not None else 0.0
            random_tensor = torch.full(simplex_shape, constant_value, device=device)
        elif random_type == "poisson":  # Poisson distribution
            lam = input_1 if input_1 is not None else 1.0
            random_tensor = torch.poisson(torch.full(simplex_shape, lam, device=device))
        elif random_type == "exponential":  # Exponential distribution
            rate = input_1 if input_1 is not None else 1.0
            random_tensor = torch.empty(simplex_shape, device=device).exponential_(rate)
        elif random_type == "beta":  # Beta distribution
            alpha = input_1 if input_1 is not None else 2.0
            beta = input_2 if input_2 is not None else 2.0
            random_tensor = torch.empty(simplex_shape, device=device).beta_(alpha, beta)
        elif random_type == "gamma":  # Gamma distribution
            shape = input_1 if input_1 is not None else 2.0
            scale = input_2 if input_2 is not None else 1.0
            random_tensor = torch.empty(simplex_shape, device=device).gamma_(shape, scale)
        else:
            raise ValueError(f"Unsupported random_type: {random_type}")

        return random_tensor

    def get_control_lambda(self, t): 
        if self.lambda_schedule == "constant":
            control_lambda = self.control_lambda

        elif self.lambda_schedule == "linear":
            control_lambda = self.control_lambda * t / self.max_timesteps

        elif self.lambda_schedule == "exp":
            control_lambda = self.control_lambda * torch.exp(t - self.max_timesteps)

        elif self.lambda_schedule == "sigmoid":
            control_lambda = self.control_lambda * torch.sigmoid(t - self.max_timesteps)

        elif self.lambda_schedule == "step":
            decay_step = 20
            decay_rate = 0.5
            control_lambda = self.control_lambda * (decay_rate ** (self.max_timesteps-t // decay_step))

        elif self.lambda_schedule == "warm_up":
            warmup_steps = 150
            control_lambda = self.control_lambda if t > warmup_steps else 1
        
        elif self.lambda_schedule == 'linear_decay':
            control_lambda = (self.lambda_alpha * self.control_lambda * t / self.max_timesteps) + self.lambda_beta
        
        elif self.lambda_schedule == 'cosine_decay':
            control_lambda = (self.lambda_alpha * self.control_lambda * 
                              (0.5 * (1 + math.cos(math.pi * t / self.max_timesteps)))) + self.lambda_beta
        else:
            control_lambda = self.control_lambda
        
        return control_lambda   
    
    def check_classifier_schedule(self, t, is_fluency_classifier=False):      
        if is_fluency_classifier:
            classifier_begin_step = self.fluency_classifier_begin_step
            classifier_end_step = self.fluency_classifier_end_step
            classifier_interval = self.fluency_classifier_interval
        else:
            classifier_begin_step = self.classifier_begin_step
            classifier_end_step = self.classifier_end_step
            classifier_interval = self.classifier_interval

        classifier_schedule = [i for i in range(classifier_begin_step, classifier_end_step, -classifier_interval)]  
            
        return (int(t) in classifier_schedule)    
    
    def apply_pos_control(self, t, logits, batch):
        control_lambda = self.get_control_lambda(t)
        true_pos_tags = batch['pos_labels']
        mask = (true_pos_tags != 3).float()
        label2id = {'[CLS]': 0, '[SEP]': 1, '[UNK]': 2, '[PAD]': 3}
        pos_lst = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB',
                'ADP', 'AUX', 'CCONJ', 'DET', 'NUM', 'PART', 'PRON', 'SCONJ',
                'PUNCT', 'SYM', 'X']
        for x in pos_lst:
            label2id[x] = len(label2id)
        id2label = {v: k for k, v in label2id.items()}

        # Store the original logits for comparison
        original_logits = logits.clone()

        for _ in range(self.control_iteration):
            with torch.enable_grad():
                logits_ctrl = logits.clone().detach().requires_grad_()
                input_prob = F.softmax(logits_ctrl, dim=-1)
                input_embeds = F.linear(input_prob, self.classifier_model.get_input_embeddings().weight.t())

                # Reshape input embeddings to [batch_size * seq_len, embed_dim]
                batch_size, seq_len, embed_dim = input_embeds.size(0), input_embeds.size(1), input_embeds.size(2)
                input_embeds_flat = input_embeds.view(batch_size * seq_len, 1, -1)

                # Pass all embeddings through the classifier at once
                pos_logits_flat = self.classifier_model(inputs_embeds=input_embeds_flat).logits

                # Reshape back to [batch_size, seq_len, num_pos_labels]
                pos_logits = pos_logits_flat.view(batch_size, seq_len, -1)
                pos_log_probs = F.log_softmax(pos_logits, dim=-1)

                # Gather log probabilities of the true POS tags across batch and sequence length
                true_pos_log_probs = pos_log_probs.gather(2, true_pos_tags.unsqueeze(-1)).squeeze(-1)

                # Compute the loss with the mask
                loss = -(true_pos_log_probs * mask).sum() / seq_len  # Divide by seq_len to average across sequence
                grad = -torch.autograd.grad(loss, logits_ctrl, retain_graph=True)[0]  # Retain graph for multiple grad calls

                logits += control_lambda * grad

            if self.eval_per_n_step > 0 and t % self.eval_per_n_step == 0:
                with torch.no_grad():
                    # Prepare logits and embeddings
                    input_prob_updated = F.softmax(logits, dim=-1)
                    input_embeds = F.linear(input_prob_updated, self.classifier_model.get_input_embeddings().weight.t())

                    # Reshape input embeddings to [batch_size * seq_len, 1, embed_dim] for single-token classification
                    batch_size, seq_len, embed_dim = input_embeds.size(0), input_embeds.size(1), input_embeds.size(2)
                    input_embeds_flat = input_embeds.view(batch_size * seq_len, 1, -1)

                    # Pass all embeddings through the classifier at once
                    pos_logits_flat = self.classifier_model(inputs_embeds=input_embeds_flat).logits

                    # Reshape back to [batch_size, seq_len, num_pos_labels]
                    pos_logits = pos_logits_flat.view(batch_size, seq_len, -1)
                    pos_log_probs = F.log_softmax(pos_logits, dim=-1)

                    # Get generated POS tags for the first batch only
                    generated_pos_tags = torch.argmax(pos_log_probs, dim=-1)[0]

                    # Decode and print for the first batch item only
                    original_decoded = self.tokenizer.decode(
                        torch.argmax(original_logits[0], dim=-1),
                        skip_special_tokens=False
                    )
                    decoded_sequence = self.tokenizer.decode(
                        torch.argmax(logits[0], dim=-1),
                        skip_special_tokens=False
                    )

                    # Convert true and generated POS tags to labels for readability
                    true_pos_labels = [id2label[idx.item()] for idx in true_pos_tags[0]]
                    generated_pos_labels = [id2label[idx.item()] for idx in generated_pos_tags]

                    # Calculate accuracy for the first batch item
                    correct_predictions = (generated_pos_tags == true_pos_tags[0]).float()
                    accuracy = correct_predictions.mean().item() * 100  # Accuracy in percentage

                    print("------------------------------------------------------")
                    print(f"Iteration: {t}")
                    print(f"Original sequence (before update): {original_decoded}")
                    print(f"Decoded sequence (after update): {decoded_sequence}")
                    print(f"True POS tags: {true_pos_labels}")
                    print(f"Generated POS tags: {generated_pos_labels}")
                    print(f"Accuracy: {accuracy:.2f}%")
                    print("------------------------------------------------------")

                    # # Print probability of the most probable token and tokens with maximum gradients
                    # cls_token_id = self.tokenizer.convert_tokens_to_ids('[CLS]')
                    # for idx in range(seq_len):
                    #     # Most probable token probability and its ID
                    #     max_prob_value, max_prob_token_id = torch.max(input_prob_updated[0, idx], dim=-1)
                    #     max_prob_token = self.tokenizer.decode([max_prob_token_id.item()])

                    #     # Maximum positive and negative gradient tokens and values
                    #     max_pos_grad_value, max_pos_token_id = torch.max(grad[0, idx], dim=-1)
                    #     max_pos_token = self.tokenizer.decode([max_pos_token_id.item()])
                        
                    #     max_neg_grad_value, max_neg_token_id = torch.min(grad[0, idx], dim=-1)
                    #     max_neg_token = self.tokenizer.decode([max_neg_token_id.item()])

                    #     # Probability and gradient for the `[CLS]` token
                    #     cls_prob_value = input_prob_updated[0, idx, cls_token_id].item()
                    #     cls_grad_value = grad[0, idx, cls_token_id].item()

                    #     # Print the details
                    #     print(f"Position {idx}:")
                    #     print(f"  Most probable token: '{max_prob_token}' (Probability: {max_prob_value.item()})")
                    #     print(f"  Max positive gradient token: '{max_pos_token}' (Gradient value: {max_pos_grad_value.item()})")
                    #     print(f"  Max negative gradient token: '{max_neg_token}' (Gradient value: {max_neg_grad_value.item()})")
                    #     print(f"  `[CLS]` token probability: {cls_prob_value}")
                    #     print(f"  `[CLS]` token gradient: {cls_grad_value}")


        return logits


    def apply_length_control(self, t, logits, batch):
        #criterion = torch.nn.CrossEntropyLoss(reduction='none')  # 16 is used to ignore certain positions

        control_lambda = self.get_control_lambda(t)
       
        target_sequence = batch['input_ids']
        span_mask = batch['span_mask']
        start_token_id = self.tokenizer.cls_token_id
        end_token_id = self.tokenizer.sep_token_id

        for _ in range(self.control_iteration):
            with torch.enable_grad():
                logits_ctrl = logits.clone().detach().requires_grad_()

                log_probs = F.log_softmax(logits_ctrl, dim=-1)

                masked_log_probs = log_probs[span_mask == 0]
                masked_target = target_sequence[span_mask == 0]

                selected_log_probs = masked_log_probs.gather(dim=-1, index=masked_target.unsqueeze(-1)).squeeze(-1)

                loss = -selected_log_probs.mean()

                grad = -torch.autograd.grad(loss, logits_ctrl)[0]

            logits += control_lambda * grad

        if self.eval_per_n_step > 0 and t % self.eval_per_n_step == 0:
            print("------------------------------------------------------")
            print(f"iteration: {t}")
            print(f"decoded: {self.tokenizer.decode(torch.argmax(logits[0], dim=-1))}")
            print(f"loss: {loss}")

            # Generate the sequences (batched)
            generated_sequence = torch.argmax(logits, dim=-1)

            # Initialize variables to track total accuracy
            total_accuracy = 0
            valid_sequences = 0
            all_target_lengths = []
            all_generated_lengths = []

            for i in range(target_sequence.size(0)):  # Iterate over each sequence in the batch
                true_start_positions = (target_sequence[i] == start_token_id).nonzero(as_tuple=True)[0]
                true_end_positions = (target_sequence[i] == end_token_id).nonzero(as_tuple=True)[0]

                if len(true_start_positions) > 0 and len(true_end_positions) > 0:
                    target_lengths = true_end_positions - true_start_positions + 1

                    try:
                        start_positions = (generated_sequence[i] == start_token_id).nonzero(as_tuple=True)[0]
                        end_positions = (generated_sequence[i] == end_token_id).nonzero(as_tuple=True)[0]

                        if len(start_positions) > 0 and len(end_positions) > 0:
                            generated_lengths = end_positions - start_positions + 1

                            # Store target and generated lengths for overall printing
                            all_target_lengths.append(target_lengths)
                            all_generated_lengths.append(generated_lengths)

                            length_condition_met = (abs(generated_lengths - target_lengths) <= 2).float()
                            accuracy = length_condition_met.mean().item()
                            total_accuracy += accuracy
                            valid_sequences += 1
                    except:
                        continue

            if valid_sequences > 0:
                average_accuracy = total_accuracy / valid_sequences
            else:
                average_accuracy = 0

            print(f"target_lengths: {all_target_lengths}")
            print(f"generated_lengths: {all_generated_lengths}")
            print(f"length_accuracy: {average_accuracy}")


        return logits

    def apply_tree_control(self, t, logits, batch):
        #criterion = torch.nn.CrossEntropyLoss(reduction='none')  # 16 is used to ignore certain positions

        control_lambda = self.get_control_lambda(t)      

        true_charts = batch['parse_chart']  # true parse tree of shape [64, 64, 124]
        mask = (true_charts != -100)
        #optimizer = optim.Adam([logits_ctrl], lr=control_lambda)

        for _ in range(self.control_iteration):
            with torch.enable_grad():
                #print(f"logits: {logits}")
                logits_ctrl = logits.clone().detach().requires_grad_()

                input_prob = F.softmax(logits_ctrl, dim=-1)
                input_embeds = F.linear(input_prob, self.classifier_model.get_input_embeddings().weight.t())

                tree_logits = self.classifier_model(inputs_embeds=input_embeds).logits
                #print(f"pos_logits: {pos_logits}")

                tree_log_probs = F.log_softmax(tree_logits, dim=-1)
                #print(f"pos_log_probs: {pos_log_probs}")

                true_charts_masked = true_charts.clone()
                true_charts_masked[~mask] = 0

                true_tree_log_probs = tree_log_probs.gather(3, true_charts_masked.unsqueeze(-1)).squeeze(-1)

                ## only consider tokens before pad tokens
                masked_loss = -(true_tree_log_probs * mask).sum()

                # total_loss = -true_tree_log_probs.sum()

                loss = masked_loss

                # loss.backward()
                # optimizer.step()

                grad = -torch.autograd.grad(loss, logits_ctrl)[0]

            logits += control_lambda * grad
            #logits = logits_ctrl.detach()

        if self.eval_per_n_step > 0 and t % self.eval_per_n_step == 0:
            print("------------------------------------------------------")
            print(f"iteration: {t}")
            print(f"decoded: {self.tokenizer.decode(torch.argmax(logits[0], dim=-1))}")
            print(f"loss: {loss}")

            # Calculate generated tree for the entire batch
            generated_tree = torch.argmax(tree_log_probs, dim=-1)

            # Flatten the tensors to make them compatible with sklearn's f1_score
            flattened_true = true_charts[mask].view(-1).cpu().numpy()
            flattened_pred = generated_tree[mask].view(-1).cpu().numpy()

            # Filter out the 0 values in both true and predicted labels
            valid_indices = (flattened_true != 0) & (flattened_pred != 0)
            valid_true = flattened_true[valid_indices]
            valid_pred = flattened_pred[valid_indices]

            # Calculate F1 score, ignoring class 0
            f1 = f1_score(valid_true, valid_pred, average='weighted')

            # Calculate total accuracy across the entire batch
            correct_predictions = (generated_tree == true_charts) & mask
            total_accuracy = correct_predictions.sum().float() / mask.sum().float()

            print(f"total_accuracy: {total_accuracy}")
            print(f"F1 Score (excluding class 0): {f1}")
            print("------------------------------------------------------")

        return logits

    def apply_span_control(self, t, logits, batch):

        control_lambda = self.get_control_lambda(t)

        true_spans = batch['spans']  # true spans of shape [batch_size, 3]: [start_idx, end_idx, syntax label]

         # Extract indices for the correct spans
        start_indices = true_spans[:, 0]
        end_indices = true_spans[:, 1]
        syntax_labels = true_spans[:, 2]

        for _ in range(self.control_iteration):
            with torch.enable_grad():
                logits_ctrl = logits.clone().detach().requires_grad_()

                input_prob = F.softmax(logits_ctrl, dim=-1)
                input_embeds = F.linear(input_prob, self.classifier_model.get_input_embeddings().weight.t())

                tree_logits = self.classifier_model(inputs_embeds=input_embeds).logits

                tree_log_probs = F.log_softmax(tree_logits, dim=-1)

                # Gather the log probabilities at the specific indices
                # Note: we gather tree logits by batch_num, start_idx, end_idx, and syntax_label
                gathered_log_probs = tree_log_probs[torch.arange(tree_logits.size(0)), start_indices, end_indices, syntax_labels]

                # Compute the loss as the negative log likelihood
                span_loss = -gathered_log_probs.mean()  # Mean over batch

                # Compute the gradient and update logits
                grad = -torch.autograd.grad(span_loss, logits_ctrl)[0]

                logits += control_lambda * grad

            if self.eval_per_n_step > 0 and t % self.eval_per_n_step == 0:
                print("------------------------------------------------------")
                print(f"iteration: {t}")
                print(f"decoded: {self.tokenizer.decode(torch.argmax(logits[0], dim=-1))}")
                print(f"loss: {span_loss}")

                # Predict the syntax labels at the specific spans
                predicted_syntax_labels = torch.argmax(tree_logits, dim=-1)
                predicted_at_spans = predicted_syntax_labels[torch.arange(tree_logits.size(0)), start_indices, end_indices]

                # Calculate accuracy only for the specific span locations
                correct_predictions = (predicted_at_spans == syntax_labels).float()
                span_accuracy = correct_predictions.mean().item()

                print(f"Span Accuracy: {span_accuracy}")
                print("------------------------------------------------------")

        return logits
    
    def apply_sentiment_control(self, t, logits, batch):
        control_lambda = self.get_control_lambda(t)
        fluency_lambda = self.fluency_classifier_lambda
        classifier_apply = self.check_classifier_schedule(t, False)
        fluency_classifier_apply = self.check_classifier_schedule(t, True)
        original_logits = logits.clone()

        if not classifier_apply and not fluency_classifier_apply:
            self.token_grad = None
            return logits        

        default_class_map = {"negative": 0, "positive": 1}
        default_class_inverse_map = {v: k for (k, v) in default_class_map.items()}
        twitter_large_class_map = {"negative": 1, "positive": 3} 
        twitter_base_class_map = {"negative": 0, "positive": 2}    
        siebert_class_map = {"negative": 0, "positive": 1}
        hartmann_class_map = {"negative": 0, "positive": 2}         

        def map_true_tags(true_tags, classifier_name):   
            # Helper function to map tags
            def get_mapped_tag(tag, class_map):
                tag_name = default_class_inverse_map[tag.item()]  # Convert ID to class name
                return class_map[tag_name]  # Map class name to classifier-specific ID

            if "twitter-roberta-base" in classifier_name:
                return torch.tensor([get_mapped_tag(tag, twitter_base_class_map) for tag in true_tags], device=true_tags.device)
            elif "twitter-roberta-large" in classifier_name:
                return torch.tensor([get_mapped_tag(tag, twitter_large_class_map) for tag in true_tags], device=true_tags.device)
            elif "siebert" in classifier_name:
                return torch.tensor([get_mapped_tag(tag, siebert_class_map) for tag in true_tags], device=true_tags.device)
            elif "hartmann" in classifier_name:
                return torch.tensor([get_mapped_tag(tag, hartmann_class_map) for tag in true_tags], device=true_tags.device)
            else:
                # Default case: return tags without remapping
                return true_tags 

        true_tags = batch['sentiment']
        input_ids = batch["input_ids"]
        span_mask = batch['span_mask']        

        # Extract input prompt (where span_mask is 0)
        input_prompt = input_ids * (~span_mask)
        #print(f"input_prompt_shape: {input_prompt.shape}")

        # Convert input prompt to embeddings
        # prompt_embeds = self.classifier_model.get_input_embeddings()(input_prompt)

        # if t > self.classifier_begin_step or t < self.classifier_end_step:
        #     return logits        
        if fluency_classifier_apply:
            true_fluency_tags = torch.ones(true_tags.size(), dtype=torch.long, device=true_tags.device)

            for idx in range(self.fluency_classifier_iteration):
                with torch.enable_grad():
                    logits_ctrl = logits.clone().detach().requires_grad_()
                    input_prob = F.softmax(logits_ctrl, dim=-1)

                    input_embeds = F.linear(input_prob, self.fluency_classifier.get_input_embeddings().weight.t())

                    fluency_logits = self.fluency_classifier(inputs_embeds=input_embeds).logits
                    fluency_log_probs = F.log_softmax(fluency_logits, dim=-1)
                    true_fluency_log_probs = fluency_log_probs.gather(1, true_fluency_tags.unsqueeze(-1)).squeeze(-1)

                    loss = -true_fluency_log_probs.sum()

                    grad = -torch.autograd.grad(loss, logits_ctrl)[0]

                logits += fluency_lambda * grad
        
        if classifier_apply:
            for idx in range(self.control_iteration):
                with torch.enable_grad():
                    logits_ctrl = logits.clone().detach().requires_grad_()     
                    input_prob = F.softmax(logits_ctrl, dim=-1)

                    if self.classifier_model is None:
                        assert self.sub_classifiers is not None, "no classifiers"

                        classifier_dict = random.choice(self.sub_classifiers)
                        classifier_name = classifier_dict['name']
                        selected_classifier = classifier_dict['classifier']

                        true_tags_mapped = map_true_tags(true_tags, classifier_name) 

                        input_embeds = F.linear(input_prob, selected_classifier.get_input_embeddings().weight.t()) 
                            
                        try:                    
                            sent_logits = selected_classifier(inputs_embeds=input_embeds).logits
                        except:
                            sent_logits = selected_classifier(inputs_embeds=input_embeds)                
                        
                        sent_log_probs = F.log_softmax(sent_logits, dim=-1)
                        true_sent_log_probs = sent_log_probs.gather(1, true_tags_mapped.unsqueeze(-1)).squeeze(-1)

                        # Mask to zero out the correct index and penalize the incorrect indexes
                        mask = torch.ones_like(sent_log_probs, dtype=torch.bool)
                        mask.scatter_(1, true_tags_mapped.unsqueeze(-1), False)
                    
                        # Sum the log probabilities for all incorrect indexes
                        false_sent_log_probs = sent_log_probs[mask].view(sent_log_probs.size(0), -1)
                        penalty = false_sent_log_probs.sum(dim=1).mean()

                        loss = -true_sent_log_probs.sum() + penalty
                        #loss = -true_sent_log_probs.sum()
                    
                    elif self.classifier_model is not None:  
                        input_embeds = F.linear(input_prob, self.classifier_model.get_input_embeddings().weight.t()) 
                            
                        try:                    
                            sent_logits = self.classifier_model(inputs_embeds=input_embeds).logits
                        except:
                            sent_logits = self.classifier_model(inputs_embeds=input_embeds)                
                        
                        sent_log_probs = F.log_softmax(sent_logits, dim=-1)
                        true_sent_log_probs = sent_log_probs.gather(1, true_tags.unsqueeze(-1)).squeeze(-1)

                        # Mask to zero out the correct index and penalize the incorrect indexes
                        mask = torch.ones_like(sent_log_probs, dtype=torch.bool)
                        mask.scatter_(1, true_tags.unsqueeze(-1), False)
                    
                        # Sum the log probabilities for all incorrect indexes
                        false_sent_log_probs = sent_log_probs[mask].view(sent_log_probs.size(0), -1)
                        penalty = false_sent_log_probs.sum(dim=1).mean()

                        loss = -true_sent_log_probs.sum() + penalty

                    if "grad" in self.token_schedule and idx==self.control_iteration-1:
                        grads = torch.autograd.grad(loss, (logits_ctrl, input_embeds))
                        grad, token_gradients = grads    
                        grad = -grad                
                        self.token_grad = token_gradients.norm(dim=-1).detach()
                    else:
                        grad = -torch.autograd.grad(loss, logits_ctrl)[0] 

                    # batch_size = grad.size(0)
                    # sequence_length = grad.size(1)

                    # # Initialize storage for token influence per class
                    # token_influence = {}

                    # # Compute gradients for each class
                    # num_classes = sent_logits.size(-1)
                    # for class_idx in range(num_classes):
                    #     self.classifier_model.zero_grad()  # Reset gradients
                    #     class_logit = sent_logits[:, class_idx]  # Logit for the current class
                    #     class_loss = class_logit.sum()  # Retain graph for next class

                    #     # Gradients w.r.t. input embeddings
                    #     token_gradients = -torch.autograd.grad(class_loss, input_embeds, retain_graph=True)[0]
                    #     token_importance = token_gradients.norm(dim=-1)  # Shape: [batch_size, seq_len]

                    #     # Influence direction: Positive gradient pushes output toward this class
                    #     influence_direction = token_gradients.sum(dim=-1)  # Shape: [batch_size, seq_len]

                    #     # Store results for the current class
                    #     token_influence[class_idx] = (token_importance, influence_direction)

                    # # Analyze influence for each token in the batch
                    # for batch_idx in range(batch_size):  # Iterate over batch
                    #     print(f"Batch {batch_idx}:")
                    #     for token_idx in range(sequence_length):  # Iterate over sequence length                        
                    #         token = self.tokenizer.decode([torch.argmax(logits_ctrl[batch_idx, token_idx], dim=-1).item()])
                    #         print(f"  Token: {token}")
                    #         for class_idx in range(num_classes):
                    #             importance, direction = token_influence[class_idx]
                    #             influence = direction[batch_idx, token_idx].item()
                    #             print(f"    Class {class_idx}: Importance: {importance[batch_idx, token_idx].item():.6f}, Influence: {influence:.6f}")

                    # print("Token-wise Gradient Details for Each Batch:")

                    # for batch_idx in range(batch_size):
                    #     # Get maximum positive and negative gradient values and their corresponding token IDs
                    #     max_pos_grad_values, max_pos_token_ids = torch.amax(grad[batch_idx], dim=-1), torch.argmax(grad[batch_idx], dim=-1)
                    #     max_neg_grad_values, max_neg_token_ids = torch.amin(grad[batch_idx], dim=-1), torch.argmin(grad[batch_idx], dim=-1)

                    #     # Decode tokens for the current batch
                    #     max_pos_tokens = self.tokenizer.batch_decode(max_pos_token_ids.tolist())
                    #     max_neg_tokens = self.tokenizer.batch_decode(max_neg_token_ids.tolist())

                    #     print(f"Batch {batch_idx}:")
                    #     for idx in range(sequence_length):
                    #         print(f"  Token Index: {idx}")
                    #         print(f"    Max Positive Gradient Value: {max_pos_grad_values[idx].item():.6f}")
                    #         print(f"    Max Positive Token: {max_pos_tokens[idx]}")
                    #         print(f"    Max Negative Gradient Value: {max_neg_grad_values[idx].item():.6f}")
                    #         print(f"    Max Negative Token: {max_neg_tokens[idx]}") 

                if self.lambda_schedule == "adaptive_logit_sequence":
                    classifier_probs = F.softmax(sent_logits, dim=-1)  # Shape: (batch_size, num_classes)
                    sequence_probs = classifier_probs.max(dim=-1).values  # Shape: (batch_size,)
                    control_lambda = (self.control_lambda * (1 - sequence_probs)).unsqueeze(-1).unsqueeze(-1)    
                elif self.lambda_schedule == "adaptive_grad_sequence":
                    grad_norm_sequence = grad.norm(p=2, dim=(1, 2))  # Shape: (batch_size,)
                    control_lambda = (self.control_lambda * (1 + grad_norm_sequence / grad_norm_sequence.mean())).unsqueeze(-1).unsqueeze(-1) 
                # elif self.lambda_schedule == "adaptive_logit_token":
                #     token_probs = F.softmax(sent_logits, dim=-1).gather(2, true_tags.unsqueeze(-1))  # Shape: (batch_size, seq_length, 1)
                #     control_lambda = (self.control_lambda * (1 - token_probs.squeeze(-1))).unsqueeze(-1)   # Shape: (batch_size, seq_length)
                # elif self.lambda_schedule == "adaptive_grad_token":
                #     grad_norm_token = grad.norm(p=2, dim=-1)  # Shape: (batch_size, seq_length)
                #     control_lambda = (self.control_lambda * (1 + grad_norm_token / grad_norm_token.mean(dim=-1, keepdim=True))).unsqueeze(-1)                   
                
                control_lambda = torch.tensor(control_lambda, device=logits.device) if not isinstance(control_lambda, torch.Tensor) else control_lambda
                control_lambda = torch.clamp(control_lambda, min=0, max=50000)
                # print(logits.shape)
                # print(control_lambda.shape)
                logits += control_lambda * grad  

        # if self.eval_per_n_step > 0 and t % self.eval_per_n_step == 0:
        #     print("------------------------------------------------------")
        #     print(f"iteration: {t}")
        #     print(f"control_lambda: {control_lambda}")
        #     # Get the predicted tokens from logits
        #     generated_tokens = torch.argmax(logits[0], dim=-1)

        #     #Combine input prompt and generated tokens using span_mask
        #     full_output = input_prompt[0] * (~span_mask[0]) + generated_tokens * span_mask[0] 

        #     print(f"decoded: {self.tokenizer.decode(torch.argmax(logits[0], dim=-1))}")
        #     print(f"full_decoded: {self.tokenizer.decode(full_output)}")
        #     print(f"loss: {loss}")

            # print("----- Token-wise Update Inspection (First Batch Item) -----")
            # topk = 3  # you can change to see top-k gradients

            # token_ids_before = torch.argmax(original_logits[0], dim=-1)
            # token_ids_after = torch.argmax(logits[0], dim=-1)
            # token_grads = grad[0].norm(dim=-1)  # shape: [seq_len]

            # for idx in range(token_ids_before.size(0)):
            #     token_before = self.tokenizer.decode([token_ids_before[idx].item()])
            #     token_after = self.tokenizer.decode([token_ids_after[idx].item()])

            #     topk_grads, topk_indices = torch.topk(grad[0][idx], topk)
            #     top_tokens = self.tokenizer.batch_decode(topk_indices.tolist())
            #     top_grad_info = ", ".join([f"{tok} ({val.item():.4f})" for tok, val in zip(top_tokens, topk_grads)])

            #     print(f"[pos {idx:02d}] before={token_before:>10}, after={token_after:>10}, top{topk} grad tokens: {top_grad_info}")
            # #############################################################
            # print("------------------------------------------------------")
            # if self.classifier_model is not None:
            #     predicted_sentiment = torch.argmax(sent_logits, dim=-1)
            #     print(f"target_sentiment: {true_tags}")
            #     print(f"predicted_sentiment: {predicted_sentiment}")
            #     correct_predictions = (predicted_sentiment == true_tags).float()
            #     sentiment_accuracy = correct_predictions.mean().item()

            #     print(f"Sentiment Accuracy: {sentiment_accuracy}")
            #     print("------------------------------------------------------")               

            # elif self.sub_classifiers is not None:
            #     for classifier_dict in self.sub_classifiers:
            #         classifier_name = classifier_dict['name']
            #         sub_classifier = classifier_dict['classifier']
            #         true_tags_mapped = map_true_tags(true_tags, classifier_name)

            #         # Generate subclassifier predictions
            #         sub_input_embeds = F.linear(F.softmax(logits, dim=-1), sub_classifier.get_input_embeddings().weight.t())
            #         try:                    
            #             sub_sent_logits = sub_classifier(inputs_embeds=sub_input_embeds).logits
            #         except:
            #             sub_sent_logits = sub_classifier(inputs_embeds=sub_input_embeds)
                    
            #         sub_predicted_sentiment = torch.argmax(sub_sent_logits, dim=-1)

            #         # Calculate accuracy for the subclassifier
            #         sub_correct_predictions = (sub_predicted_sentiment == true_tags_mapped).float()
            #         sub_sentiment_accuracy = sub_correct_predictions.mean().item()
            #         print(f"Subclassifier ({classifier_name}) Topic Accuracy: {sub_sentiment_accuracy}")
            #         print("------------------------------------------------------") 
        return logits

    def apply_topic_control(self, t, logits, batch):
        control_lambda = self.get_control_lambda(t)
        classifier_apply = self.check_classifier_schedule(t)

        if not classifier_apply:
            self.token_grad = None
            return logits      

        true_tags = batch['topic']

        default_class_map = {"world": 0, "sports": 1, "business": 2, "sci-tech": 3}
        default_class_inverse_map = {v: k for (k, v) in default_class_map.items()}
        single_class_map = {"world": 1, "sports": 1, "business": 1, "sci-tech": 1}
        twitter_class_map = {"sports": 16, "business": 1, "sci-tech": 15}        
        nyt_class_map = {"sports": 0, "business": 2, "sci-tech": 5}        
        agnews_class_map = {"world": 0, "sports": 1, "business": 2, "sci-tech": 3}
        

        def map_true_tags(true_tags, classifier_name):           

            # Helper function to map tags
            def get_mapped_tag(tag, class_map):
                tag_name = default_class_inverse_map[tag.item()]  # Convert ID to class name
                return class_map[tag_name]  # Map class name to classifier-specific ID

            if "twitter" in classifier_name:
                return torch.tensor([get_mapped_tag(tag, twitter_class_map) for tag in true_tags], device=true_tags.device)
            elif "nyt" in classifier_name:
                return torch.tensor([get_mapped_tag(tag, nyt_class_map) for tag in true_tags], device=true_tags.device)
            elif "agnews" in classifier_name:
                return torch.tensor([get_mapped_tag(tag, agnews_class_map) for tag in true_tags], device=true_tags.device)
            elif "output" in classifier_name:
                return torch.tensor([get_mapped_tag(tag, single_class_map) for tag in true_tags], device=true_tags.device)
            else:
                # Default case: return tags without remapping
                return true_tags

        for idx in range(self.control_iteration):
            with torch.enable_grad():
                logits_ctrl = logits.clone().detach().requires_grad_()
                input_prob = F.softmax(logits_ctrl, dim=-1)       

                if self.classifier_model is None:
                    assert self.sub_classifiers is not None, "no classifiers"

                    classifier_dict = random.choice(self.sub_classifiers)
                    classifier_name = classifier_dict['name']
                    selected_classifier = classifier_dict['classifier']

                    true_tags_mapped = map_true_tags(true_tags, classifier_name) 

                    input_embeds = F.linear(input_prob, selected_classifier.get_input_embeddings().weight.t())
                    sent_logits = selected_classifier(inputs_embeds=input_embeds).logits

                    sent_log_probs = F.log_softmax(sent_logits, dim=-1)  
                    true_sent_log_probs = sent_log_probs.gather(1, true_tags_mapped.unsqueeze(-1)).squeeze(-1)

                    # Mask to zero out the correct index and penalize the incorrect indexes
                    mask = torch.ones_like(sent_log_probs, dtype=torch.bool)
                    mask.scatter_(1, true_tags_mapped.unsqueeze(-1), False)
                
                    # Sum the log probabilities for all incorrect indexes
                    false_sent_log_probs = sent_log_probs[mask].view(sent_log_probs.size(0), -1)
                    penalty = false_sent_log_probs.sum(dim=1).mean()

                    loss = -true_sent_log_probs.sum() + penalty 

                elif self.classifier_model is not None:  
                    input_embeds = F.linear(input_prob, self.classifier_model.get_input_embeddings().weight.t())
                    sent_logits = self.classifier_model(inputs_embeds=input_embeds).logits
                    #sent_logits = self.classifier_model(inputs_embeds=input_embeds)                
                
                    sent_log_probs = F.log_softmax(sent_logits, dim=-1)  
                    true_tags_mapped = map_true_tags(true_tags, "output")
                    true_sent_log_probs = sent_log_probs.gather(1, true_tags_mapped.unsqueeze(-1)).squeeze(-1)

                    # Mask to zero out the correct index and penalize the incorrect indexes
                    mask = torch.ones_like(sent_log_probs, dtype=torch.bool)
                    mask.scatter_(1, true_tags_mapped.unsqueeze(-1), False)                   
                
                    # Sum the log probabilities for all incorrect indexes
                    false_sent_log_probs = sent_log_probs[mask].view(sent_log_probs.size(0), -1)
                    penalty = false_sent_log_probs.sum(dim=1).mean()

                    loss = -true_sent_log_probs.sum() + penalty 

                    if self.sub_classifiers is not None:
                        total_sub_loss = 0
                        for classifier_dict in self.sub_classifiers:
                            classifier_name = classifier_dict['name']
                            sub_classifier = classifier_dict['classifier']
                            # Map true tags to the current subclassifier's label space
                            true_tags_mapped = map_true_tags(true_tags, classifier_name)
                            
                            # Compute logits for the subclassifier
                            sub_input_embeds = F.linear(input_prob, sub_classifier.get_input_embeddings().weight.t())
                            sub_sent_logits = sub_classifier(inputs_embeds=sub_input_embeds).logits
                            sub_sent_log_probs = F.log_softmax(sub_sent_logits, dim=-1)
                            
                            # Calculate the log probabilities of the true labels
                            true_sub_sent_log_probs = sub_sent_log_probs.gather(1, true_tags_mapped.unsqueeze(-1)).squeeze(-1)
                            
                            # Calculate penalty for incorrect predictions
                            sub_mask = torch.ones_like(sub_sent_log_probs, dtype=torch.bool)
                            sub_mask.scatter_(1, true_tags_mapped.unsqueeze(-1), False)
                            
                            false_sub_sent_log_probs = sub_sent_log_probs[sub_mask].view(sub_sent_log_probs.size(0), -1)
                            sub_penalty = false_sub_sent_log_probs.sum(dim=1).mean()
                            
                            # Accumulate the loss for the subclassifier
                            sub_loss = -true_sub_sent_log_probs.sum() + sub_penalty
                            total_sub_loss += sub_loss
                        
                        total_sub_loss = total_sub_loss / len(self.sub_classifiers)
                        loss += total_sub_loss


                # if self.sub_classifier is not None:
                #     assert self.classifier_model is not None
                #     sub_input_embeds = F.linear(input_prob, self.sub_classifier.get_input_embeddings().weight.t())
                #     sub_sent_logits = self.sub_classifier(inputs_embeds=sub_input_embeds).logits
                #     sub_sent_log_probs = F.log_softmax(sub_sent_logits, dim=-1)
                #     false_sub_sent_log_probs = sub_sent_log_probs.gather(1, (1-true_tags).unsqueeze(-1)).squeeze(-1)       

                #     loss = -0.6*true_sent_log_probs.sum() - 0.4*false_sub_sent_log_probs.sum()
                # elif self.sub_classifiers is not None:
                #     # Multiple sub-classifiers case
                #     false_sent_log_probs = []

                #     for sub_classifier in self.sub_classifiers:
                #         sub_input_embeds = F.linear(input_prob, sub_classifier.get_input_embeddings().weight.t())
                #         sub_sent_logits = sub_classifier(inputs_embeds=sub_input_embeds).logits
                #         sub_sent_log_probs = F.log_softmax(sub_sent_logits, dim=-1)
                #         false_sub_sent_log_probs = sub_sent_log_probs.gather(1, (1 - true_tags).unsqueeze(-1)).squeeze(-1)
                #         false_sent_log_probs.append(false_sub_sent_log_probs)

                #     # Stack all false log probabilities and take the mean
                #     mean_false_log_probs = torch.stack(false_sent_log_probs, dim=0).mean(dim=0)

                #     # Combine true and false log probabilities into the final loss
                #     loss = -0.6*true_sent_log_probs.sum() - 0.4*mean_false_log_probs.sum()   

                if "grad" in self.token_schedule and idx==self.control_iteration-1:
                    grads = torch.autograd.grad(loss, (logits_ctrl, input_embeds))
                    grad, token_gradients = grads    
                    grad = -grad                
                    self.token_grad = token_gradients.norm(dim=-1).detach()
                else:
                    grad = -torch.autograd.grad(loss, logits_ctrl)[0]

                #grad = -torch.autograd.grad(loss, logits_ctrl)[0]

            logits += control_lambda * grad

        if self.eval_per_n_step > 0 and t % self.eval_per_n_step == 0:
            print("------------------------------------------------------")
            print(f"Iteration: {t}")
            print(f"Decoded: {self.tokenizer.decode(torch.argmax(logits[0], dim=-1))}")
            print(f"Loss: {loss}")

            # Evaluate the main classifier if available
            if self.classifier_model is not None:
                true_tags_mapped = map_true_tags(true_tags, "output")
                predicted_topic = torch.argmax(sent_logits, dim=-1)
                correct_predictions = (predicted_topic == true_tags_mapped).float()
                topic_accuracy = correct_predictions.mean().item()
                print(f"Main Classifier Topic Accuracy: {topic_accuracy}")

            # Evaluate each subclassifier if available
            if self.sub_classifiers is not None:
                for classifier_dict in self.sub_classifiers:
                    classifier_name = classifier_dict['name']
                    sub_classifier = classifier_dict['classifier']
                    true_tags_mapped = map_true_tags(true_tags, classifier_name)

                    # Generate subclassifier predictions
                    sub_input_embeds = F.linear(F.softmax(logits, dim=-1), sub_classifier.get_input_embeddings().weight.t())
                    sub_sent_logits = sub_classifier(inputs_embeds=sub_input_embeds).logits
                    sub_predicted_topic = torch.argmax(sub_sent_logits, dim=-1)

                    # Calculate accuracy for the subclassifier
                    sub_correct_predictions = (sub_predicted_topic == true_tags_mapped).float()
                    sub_topic_accuracy = sub_correct_predictions.mean().item()
                    print(f"Subclassifier ({classifier_name}) Topic Accuracy: {sub_topic_accuracy}")

            print("------------------------------------------------------")

        return logits
    
    def apply_toxicity_control(self, t, logits, batch):
        control_lambda = self.get_control_lambda(t)
        classifier_apply = self.check_classifier_schedule(t)

        if not classifier_apply:
            self.token_grad = None
            return logits      

        true_tags = batch['toxicity']

        for idx in range(self.control_iteration):
            with torch.enable_grad():
                logits_ctrl = logits.clone().detach().requires_grad_()

                input_prob = F.softmax(logits_ctrl, dim=-1)
                input_embeds = F.linear(input_prob, self.classifier_model.get_input_embeddings().weight.t())
                #print(f"input_embeds_shape: {input_embeds.shape}")

                #sent_logits = self.classifier_model(inputs_embeds=input_embeds).logits
                sent_logits = self.classifier_model(inputs_embeds=input_embeds).logits
                sent_log_probs = F.log_softmax(sent_logits, dim=-1)

                true_sent_log_probs = sent_log_probs[:, 0]

                loss = -true_sent_log_probs.sum()

                if "grad" in self.token_schedule and idx==self.control_iteration-1:
                    grads = torch.autograd.grad(loss, (logits_ctrl, input_embeds))
                    grad, token_gradients = grads    
                    grad = -grad                
                    self.token_grad = token_gradients.norm(dim=-1).detach()
                else:
                    grad = -torch.autograd.grad(loss, logits_ctrl)[0] 

            logits += control_lambda * grad

        if self.eval_per_n_step > 0 and t % self.eval_per_n_step == 0:
            print("------------------------------------------------------")
            print(f"iteration: {t}")
            print(f"decoded: {self.tokenizer.decode(torch.argmax(logits[0], dim=-1))}")
            print(f"loss: {loss}")

            predicted_sentiment = torch.argmax(sent_logits, dim=-1)
            correct_predictions = (predicted_sentiment == true_tags).float()
            sentiment_accuracy = correct_predictions.mean().item()
            
            # prob of toxic sentence
            toxicity_log_probs = sent_log_probs[:, 1]
            toxicity_probs = toxicity_log_probs.exp()
            
            print(f"Average toxicity: {toxicity_probs.mean()}")

            print(f"Toxicity Accuracy: {sentiment_accuracy}")
            print("------------------------------------------------------")

        return logits  
    




