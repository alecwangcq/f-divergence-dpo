from trlx.trainer.accelerate_ppo_trainer import AcceleratePPOTrainer
import os
from time import time
from typing import Callable, List

import json
import torch
import torch.nn.functional as F

import trlx.utils.logging as logging
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.ppo_types import PPORLElement, PPORLBatch

from trlx.utils import Clock
from trlx.utils.modeling import gather_dict, logprobs_of_labels
from trlx.trainer import register_trainer

logger = logging.get_logger(__name__)


@register_trainer
class AccelerateFPPOTrainer(AcceleratePPOTrainer):

    def setup_model(self):
        model = super().setup_model()
        
        if self.config.model.archive is not None:
            print(f'Loading reference model from {self.config.model.archive}...')
            state_dict = torch.load(self.config.model.archive, map_location='cpu')
            step, metrics = state_dict['step_idx'], state_dict['metrics']
            print(f'loading pre-trained weights at step {step} from {self.config.model.archive} with metrics {json.dumps(metrics, indent=2)}')
            # import ipdb; ipdb.set_trace()
            model.base_model.load_state_dict(state_dict['state'])
        return model
    
    # def setup_model(self):
    #     """
    #     Returns a model derived from an instance's TRLConfig
    #     """
    #     logger.info(f"Initializing model: {self.config.model.model_path}")

    #     # Retrieves model equipped for ppo, ilql, etc
    #     model = self.get_arch(self.config)

    #     if self.config.model.peft_config is None:
    #         if self.config.model.model_arch_type == "seq2seq":
    #             freeze_bottom_seq2seq_layers(model.base_model, self.config.model.num_layers_unfrozen)
    #         else:
    #             freeze_bottom_causal_layers(model.base_model, self.config.model.num_layers_unfrozen)
    #     else:
    #         if self.accelerator.is_main_process and hasattr(model.base_model, "print_trainable_parameters"):
    #             model.base_model.print_trainable_parameters()
    #         if self.config.model.num_layers_unfrozen >= 0:
    #             logger.warning(
    #                 "The argument num_layers_unfrozen is ignored when using peft, to prevent unexpected behaviour."
    #                 "For Lora, use the `LoraConfig` argument `modules_to_save` instead."
    #             )

    #     return model
    
    def loss(self, batch: PPORLBatch):
        """Forward pass & loss

        Args:
            batch: Previous batch of episodes
        """
        # Move `batch` data to `accelerator` device
        query_tensors = batch.query_tensors.to(self.accelerator.device)
        response_tensors = batch.response_tensors.to(self.accelerator.device)
        old_logprobs = batch.logprobs.to(self.accelerator.device)
        old_values = batch.values.to(self.accelerator.device)
        old_rewards = batch.rewards.to(self.accelerator.device)
        response_length = old_rewards.shape[1]

        advantages, returns = self.config.method.get_advantages_and_returns(old_values, old_rewards, response_length)

        tokens = torch.cat((query_tensors, response_tensors), dim=1)
        attention_mask = tokens.not_equal(self.tokenizer.pad_token_id).long().to(tokens.device)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        outputs = self.model(tokens, attention_mask, return_dict=True, position_ids=position_ids)
        with torch.no_grad():
            ref_logits = self.model.forward_hydra(tokens, attention_mask=attention_mask, position_ids=position_ids, return_dict=True).logits
            ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], tokens[:, 1:])
        logits = outputs.logits
        values_pred = outputs.value
        values_pred = values_pred[:, :-1]
        logprobs = logprobs_of_labels(logits[:, :-1, :], tokens[:, 1:])
        
        start = query_tensors.shape[1] - 1
        end = start + response_length
        logprobs, values_pred, mask = (
                logprobs[:, start:end],
                values_pred[:, start:end],
                attention_mask[:, start + 1 : end + 1],
            )
        ref_logprobs = ref_logprobs[:, start:end]
        log_ratio = (logprobs - ref_logprobs) * mask  # log q/p
        ratio = torch.exp(-log_ratio)  # p/q
        f_ratio = self.config.method.f(ratio)  # f(p/q)
        kl = f_ratio - self.config.method.f_prime_one * (ratio - 1)
        kl /= (mask.sum(dim=-1, keepdim=True) + 1)
        kl_loss = self.kl_ctl.value * kl.sum(dim=-1).mean()
        loss, stats = self.config.method.loss(
            logprobs=logprobs,
            values=values_pred,
            old_logprobs=old_logprobs,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            mask=mask,
        )
        if not self.config.method.kl_in_reward:  # 'not' means 'kl_in_loss'
            loss = loss + kl_loss            
        return loss, stats
    
    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):  # noqa:
        """Make experiences

        Takes `chunk_size` number of prompts from `prompt_iterator`, samples
        from the model and then computes the KL against a reference model. Finally it
        then appends PPOElements to trainer's `store`.

        Args:
            num_rollouts: Number of rollouts to generate
            iter_count: Total number of updates run (i.e. number of updates run for all batches & epochs)
        """
        logger.info("Collecting rollouts")
        tbar = logging.tqdm(
            total=num_rollouts,
            disable=os.environ.get("RANK", 0) != "0",
            desc=f"[rollout 0 / {num_rollouts}]",
            # Lower progress bar by 1 if we're in WARNING mode or above to avoid hiding high priority progress
            # bars (e.g. loss progress in trainers)
            position=logging.get_verbosity() >= logging.WARNING,
            # Leave progress bar if we're in INFO mode or lower to avoid spamming in suppressed verbosity levels
            leave=logging.get_verbosity() < logging.WARNING,
        )

        clock = Clock()
        ppo_rl_elements = []
        accumulated_stats = []

        while len(ppo_rl_elements) < num_rollouts:
            stats = {}
            # Get next batch in prompt dataset
            batch: PromptBatch = next(self.prompt_iterator)

            rollout_generate_time = time()

            # Generate samples from the language model (similar to using HuggingFace `generate` method)
            samples = self.generate(batch["input_ids"], batch["attention_mask"])
            stats["time/rollout_generate"] = time() - rollout_generate_time

            prompt_tensors = batch.input_ids
            device = samples.device

            prompt_sizes = torch.tensor([prompt_tensors.shape[1]] * len(prompt_tensors), device=device)
            padded_samples = self.accelerator.pad_across_processes(
                samples, dim=1, pad_index=self.tokenizer.eos_token_id, pad_first=False
            )
            padded_prompts = self.accelerator.pad_across_processes(
                prompt_tensors, dim=1, pad_index=self.tokenizer.eos_token_id, pad_first=False
            )
            gathered_samples = self.accelerator.gather(padded_samples)
            gathered_prompts = self.accelerator.gather(padded_prompts)
            gathered_prompt_sizes = self.accelerator.gather(prompt_sizes)
            metadata = gather_dict({k: v for k, v in batch.items() if k != "input_ids" and k != "attention_mask"})

            if self.accelerator.is_main_process:
                all_str_samples, all_str_prompts, all_str_outputs = self.decode(
                    gathered_prompts, gathered_samples, gathered_prompt_sizes, append_eos_token=True
                )

                rollout_score_time = time()
                all_scores = torch.tensor(
                    self.reward_fn(
                        samples=all_str_samples, prompts=all_str_prompts, outputs=all_str_outputs, **metadata
                    ),
                    dtype=torch.float,
                    device=device,
                )
                stats["time/rollout_score"] = time() - rollout_score_time

                all_scores = list(all_scores.reshape(self.accelerator.num_processes, -1).unbind())
            else:
                all_scores = None

            if torch.distributed.is_initialized():
                scores = torch.empty(len(samples), device=device)
                torch.distributed.scatter(scores, all_scores)
            else:
                scores = all_scores[0].clone().detach()

            str_samples, str_prompts, str_outputs = self.decode(prompt_tensors, samples, append_eos_token=True)

            # Pad the sample outputs
            outputs = self.tokenizer(str_outputs).input_ids
            if self.config.model.model_arch_type == "seq2seq":
                # add <pad> to the start of the output
                for i in range(len(outputs)):
                    outputs[i] = [self.tokenizer.pad_token_id] + outputs[i]

            outputs = list(map(torch.LongTensor, outputs))
            maxsize = max(map(len, outputs))
            outputs = [
                F.pad(
                    output,
                    (0, maxsize - len(output)),
                    value=self.tokenizer.pad_token_id,
                )
                for output in outputs
            ]
            sample_outputs = torch.vstack(outputs).to(device)

            if self.config.method.cliprange_reward:
                scores = torch.clip(scores, -self.config.method.cliprange_reward, self.config.method.cliprange_reward)

            # store statistics of the initial rollout as reference
            if self.ref_mean is None:
                self.ref_mean, self.ref_std = scores.mean(), scores.std()
            all_scores_mean, all_scores_std = self.running_moments.update(scores)
            stats["rollout_scores/mean"] = all_scores_mean.item()
            stats["rollout_scores/std"] = all_scores_std.item()
            stats["rollout_scores/running_mean"] = self.running_moments.mean.item()
            stats["rollout_scores/running_std"] = self.running_moments.std.item()

            if self.config.method.scale_reward == "running":
                scores /= self.running_moments.std
            elif self.config.method.scale_reward == "ref":
                scores /= self.ref_std

            if True:
                all_tokens = torch.cat((prompt_tensors.to(device), sample_outputs), dim=1)
                attention_mask = all_tokens.not_equal(self.tokenizer.pad_token_id).long().to(device)
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                # with torch.no_grad():
                with torch.no_grad():
                    logits, *_, values = self.model(
                        all_tokens, attention_mask=attention_mask, position_ids=position_ids
                    )
                    if hasattr(self.model, "frozen_head") or self.model.peft_type:
                        ref_logits = self.model.forward_hydra(all_tokens, attention_mask=attention_mask, position_ids=position_ids, return_dict=True, ).logits
                    else:
                        ref_logits = self.ref_model(
                                all_tokens,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                return_dict=True,
                            ).logits
                        ref_logits = ref_logits.to(device)

            if True:
                logprobs = logprobs_of_labels(logits[:, :-1, :], all_tokens[:, 1:])
                ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], all_tokens[:, 1:])

            n_samples: int = samples.shape[0]

            # Estimate the KL divergence between the model and reference model
            if self.config.model.model_arch_type == "seq2seq":
                attention_mask = sample_outputs != self.tokenizer.pad_token_id
                start = 0
            else:
                start = prompt_tensors.shape[1] - 1
                
            # compute entropy
            # import ipdb; ipdb.set_trace()
            probabilities = F.softmax(logits[:, :-1, :], dim=-1)
            entropy = - (torch.sum(probabilities * torch.log(probabilities), dim=-1) * attention_mask[:, :-1]).mean()
            stats["policy/predictive_entropy"] = entropy.item()

            # compute f divergence
            log_ratio = (logprobs - ref_logprobs) * attention_mask[:, :-1]  # log q/p
            ratio = torch.exp(-log_ratio)  # p/q
            f_ratio = self.config.method.f(ratio)  # f(p/q)
            kl = f_ratio - self.config.method.f_prime_one * (ratio - 1)
            # print(kl.mean(), kl.min(), kl.max())
            mean_kl_per_token = kl.mean()
            mean_kl = kl.sum(1).mean()
            # import ipdb; ipdb.set_trace()
            logprobs = logprobs.cpu()
            ref_logprobs = ref_logprobs.cpu()
            prompt_tensors = prompt_tensors.cpu()
            sample_outputs = sample_outputs.cpu()
            values = values.cpu()[:, :-1]

            # Get the logprobs and values, for tokens that are not padding,
            # from the start of the prompt up to the <eos> token, while also including the latter
            # (these are taken from the student model and not the reference model)
            ends = start + attention_mask[:, start:].sum(1) + 1
            all_values = [values[ix, start : ends[ix]] for ix in range(n_samples)]
            all_logprobs = [logprobs[ix, start : ends[ix]] for ix in range(n_samples)]
            
            # Compute the KL penalty for each sample
            # kl_penalty = self.kl_ctl.value * -log_ratio.cpu()  # log p/q
            kl_penalty = self.kl_ctl.value * -f_ratio # -f(p/q) 
            kl_penalty = [xs[start : ends[ix]] for ix, xs in enumerate(kl_penalty)]
            
            # mean_kl_penalty_loss = torch.cat(kl_penalty).mean()
            # self.mean_kl_penalty_loss = mean_kl_penalty_loss

            rollout_count = 0
            
            _coef = 1. if self.config.method.kl_in_reward else 0.
            for sample_idx in range(n_samples):
                rewards = kl_penalty[sample_idx].detach().cpu() * _coef
                rewards[-1] += scores[sample_idx].cpu()

                ppo_rl_elements.append(
                    PPORLElement(
                        query_tensor=prompt_tensors[sample_idx],
                        response_tensor=sample_outputs[sample_idx],
                        logprobs=all_logprobs[sample_idx],
                        values=all_values[sample_idx],
                        rewards=rewards,
                    )
                )

                rollout_count += 1
            # import ipdb; ipdb.set_trace()
            # print(rewards)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(mean_kl, torch.distributed.ReduceOp.AVG)

            stats["time/rollout_time"] = clock.tick()
            stats["policy/sqrt_kl"] = torch.sqrt(mean_kl).item()
            stats["policy/kl_per_token"] = torch.sqrt(mean_kl_per_token).item()
            accumulated_stats.append(stats)

            tbar.set_description(f"[rollout {len(ppo_rl_elements)} / {num_rollouts}]")
            tbar.update(min(rollout_count, num_rollouts))
        tbar.close()

        stats = {k: sum([xs[k] for xs in accumulated_stats]) / len(accumulated_stats) for k in stats}
        stats["kl_ctl_value"] = self.kl_ctl.value
        self.mean_kl = stats["policy/sqrt_kl"] ** 2
        self.accelerator.log(stats, step=iter_count)

        # Push samples and rewards to trainer's rollout storage
        self.push_to_store(ppo_rl_elements)