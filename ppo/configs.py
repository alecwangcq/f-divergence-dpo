from pathlib import Path
from dataclasses import dataclass
import torch
from torchtyping import TensorType

from trlx.data.method_configs import register_method
from trlx.models.modeling_ppo import PPOConfig

from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

from trlx.utils.modeling import get_tensor_stats, flatten_dict

# @dataclass
# @register_method
# class FPPOConfig(PPOConfig):
#     def loss(
#         self,
#         logprobs: TensorType["batch_size", "response_size"],
#         values: TensorType["batch_size", "response_size"],
#         old_logprobs: TensorType["batch_size", "response_size"],
#         old_values: TensorType["batch_size", "response_size"],
#         advantages: TensorType["batch_size", "response_size"],
#         returns: TensorType["batch_size", "response_size"],
#         mask: TensorType["batch_size", "response_size"],
#     ):
#         """F-PPO objective function.
#         References:
#         - https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
#         """
#         values_clipped = torch.clamp(
#             values,
#             old_values - self.cliprange_value,
#             old_values + self.cliprange_value,
#         )
#         n = mask.sum()

#         vf_loss1 = (values - returns) ** 2
#         vf_loss2 = (values_clipped - returns) ** 2
#         vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / n

#         vf_clipfrac = torch.sum((vf_loss2 > vf_loss1).float() * mask) / n

#         log_ratio = (logprobs - old_logprobs) * mask
#         ratio = torch.exp(-log_ratio)
#         # Unbiased KL-div estimates (`k3`). Ref: http://joschu.net/blog/kl-approx.html
        
#         # TODO: implement for f-divergence, the current one is only for reverse KL divergence, i.e., KL(q||p)
#         # To implement for f-divergence, f(ratio) - f'(1) * (ratio-1).
#         with torch.no_grad():
#             # approx_kl = torch.mean((ratio - 1) - log_ratio)
#             approx_f_div = torch.mean(self.f(ratio) - self.f_prime_one * (ratio - 1))

#         pg_loss1 = -advantages * ratio
#         pg_loss2 = -advantages * torch.clamp(
#             ratio,
#             1.0 - self.cliprange,
#             1.0 + self.cliprange,
#         )
#         pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / n
#         pg_clipfrac = torch.sum((pg_loss2 > pg_loss1).float() * mask) / n

#         loss = pg_loss + self.vf_coef * vf_loss

#         stats = dict(
#             losses=dict(
#                 total_loss=loss.item(),
#                 policy_loss=pg_loss.item(),
#                 value_loss=vf_loss.item(),
#             ),
#             values=dict(
#                 get_tensor_stats(values, mask, n),
#                 values_error=torch.sum(((values - returns) * mask) ** 2) / n,
#                 values_mape_error=torch.sum((abs(values - returns) * mask) / abs(returns * mask + 1e-2)) / n,
#                 clipfrac=vf_clipfrac,
#             ),
#             old_values=get_tensor_stats(old_values, mask, n),
#             returns=get_tensor_stats(returns, mask, n),
#             policy=dict(approx_kl=approx_f_div.item(), clipfrac=pg_clipfrac.item()),
#             ratio=(ratio * mask).sum() / n,
#             padding_percentage=1 - n / mask.numel(),
#         )

#         return loss, flatten_dict(stats)

def get_f_divergence(f_divergence, alpha=0.5):
    if f_divergence == 'reverse_kl':
        f = lambda x: -torch.log(x)
        f_prime_one = -1.
    elif f_divergence == 'forward_kl':
        f = lambda x: x * torch.log(x)
        f_prime_one = 1.
    elif f_divergence == 'jsd':
        f = lambda x: x * torch.log(x) - (x+1) * torch.log((x+1)/2)
        f_prime_one = 0. 
    elif f_divergence == 'alpha_divergence':
        assert alpha > 0, 'alpha must be positive'
        assert alpha < 1, 'alpha must be less than 1'
        f = lambda x: (1.0 / (alpha * (alpha - 1))) * (torch.pow(x, alpha) - alpha * x + alpha - 1)
        f_prime_one = 0.
    else:
        raise NotImplementedError
    return f, f_prime_one

def default_fppo_config(f_divergence, alpha, init_kl_coef):
    if f_divergence == "alpha_divergence":
        tags = [f_divergence, str(alpha), f"init_kl_coef_{init_kl_coef}"]
    else:
        tags = [f_divergence, f"init_kl_coef_{init_kl_coef}"]
    method= PPOConfig(
            name="PPOConfig",
            num_rollouts=128,
            chunk_size=128,
            ppo_epochs=4,
            # init_kl_coef=0.001,
            init_kl_coef= init_kl_coef,  # forward kl 0.000001, jsd:  0.0001
            # target=3.0,
            target=None,
            horizon=10000,
            gamma=1.0,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=1,
            scale_reward="running",
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
            gen_kwargs=dict(
                max_new_tokens=60,
                top_k=0,
                top_p=1.0,
                do_sample=True,
            ),
        )
    return TRLConfig(
        train=TrainConfig(
            seq_length=1024,
            epochs=100,
            total_steps=10000,
            batch_size=32,
            checkpoint_interval=10000,
            eval_interval=50,
            pipeline="PromptPipeline",
            trainer="AccelerateFPPOTrainer",
            tags = tags
        ),
        model=ModelConfig(model_path="lvwerra/gpt2-imdb", num_layers_unfrozen=2),
        tokenizer=TokenizerConfig(tokenizer_path="gpt2", truncation_side="right"),
        optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=3e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1e-6)
        ),
        scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=3e-5)),
        method=method
    )
