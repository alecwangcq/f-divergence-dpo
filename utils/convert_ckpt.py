import sys
sys.path.append('/home/chaoqi/projects/repos/direct-preference-optimization')
import torch
from ppo.trainer import AccelerateFPPOTrainer
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.utils.loading import get_trainer

default_config = TRLConfig(
    train=TrainConfig(
        seq_length=1024,
        epochs=10000,
        total_steps=10000,
        batch_size=4,
        checkpoint_interval=10000,
        eval_interval=500,
        pipeline="PromptPipeline",
        trainer="AccelerateFPPOTrainer",
        checkpoint_dir="checkpoints/debug",
    ),
    model=ModelConfig(model_path="EleutherAI/pythia-2.8b"),
    tokenizer=TokenizerConfig(tokenizer_path="EleutherAI/pythia-2.8b", truncation_side="left"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=8e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=8e-6)),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=64,
        chunk_size=16,
        ppo_epochs=4,
        init_kl_coef=0.05,
        target=6,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=1,
        scale_reward="running",
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        
        gen_kwargs=dict(
            max_new_tokens=128,
            top_k=0,
            top_p=1.0,
            do_sample=True,
        ),
    ),
)

# name = 'ppo_hh_jsd_0.1'
name = 'ppo_hh_forward_kl_0.1'
#name = 'ppo_hh_reverse_kl_0.1'

# jsd
# cckpt_file = '/net/scratch/chaoqi/repos/direct-preference-optimization/ppo/checkpoints/ppo_hh_jsd_0.1/best_checkpoint/pytorch_model/mp_rank_00_model_states.pt'
# fwd 
ckpt_file = f'/net/scratch/chaoqi/repos/direct-preference-optimization/ppo/checkpoints/{name}/best_checkpoint/pytorch_model/mp_rank_00_model_states.pt'
# rev
# ckpt_file = '/net/scratch/chaoqi/repos/direct-preference-optimization/ppo/checkpoints/ppo_hh_reverse_kl_0.1/best_checkpoint/pytorch_model/mp_rank_00_model_states.pt'


ignores = ["v_head.0.weight", "v_head.0.bias", "v_head.2.weight", "v_head.2.bias"]
def revise_state_dict(state):
    new_state = {}
    for k, v in state.items():
        if k in ignores:
            new_state[k] = v
        else:
            new_state['base_model.'+k] = v
    return new_state

default_config.model.archive=None

trainer = get_trainer(default_config.train.trainer)(default_config)
print(trainer.accelerator._models[0])
state_dict = torch.load(ckpt_file, map_location='cpu')['module']
new_state_dict = revise_state_dict(state_dict)
try:
    trainer.accelerator._models[0].load_state_dict(new_state_dict)
except:
    import ipdb; ipdb.set_trace()
    
try:
    trainer.accelerator._models[0].save_pretrained(f'/home/chaoqi/projects/weights/temp/checkpoint/{name}')
except:
    import ipdb; ipdb.set_trace()

# import ipdb; ipdb.set_trace()
# trainer.load('/home/chaoqi/projects/repos/direct-preference-optimization/ppo/checkpoints/ppo_hh_forward_kl_0.1/best_checkpoint')
