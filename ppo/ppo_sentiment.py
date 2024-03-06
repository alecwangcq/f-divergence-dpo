# Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# with a sentiment reward function
import argparse
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from typing import List

import torch
from datasets import load_dataset
from transformers import pipeline

import trlx
from trlx.data.default_configs import TRLConfig
from configs import get_f_divergence, default_fppo_config
from trainer import AccelerateFPPOTrainer


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


def main(hparams={}, additional_hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_fppo_config(additional_hparams['f_divergence'], additional_hparams['alpha'], additional_hparams['init_kl_coef']).to_dict(), hparams)
    print(config)
    f, f_prime_one = get_f_divergence(additional_hparams.get('f_divergence', 'reverse_kl'), additional_hparams.get('alpha', 0.5))
    config.method.f = f
    config.method.f_prime_one = f_prime_one
    config.method.kl_in_reward = additional_hparams.get('kl_in_reward', False)
    # print(config)
    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    sentiment_fn = pipeline(
        "sentiment-analysis",
        "siebert/sentiment-roberta-large-english",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=device,
    )

    def reward_fn(samples: List[str], **kwargs) -> List[float]:
        sentiments = list(map(get_positive_score, sentiment_fn(samples)))
        return sentiments

    # Use the test split for evaluation
    imdb_test = load_dataset("imdb", split="test")
    eval_prompts = [" ".join(review.split()[:4]) for review in imdb_test["text"]]
    # eval_prompts = ["I don't know much about Hungarian underground"] * 256

    # Use the train split for training
    imdb_train = load_dataset("imdb", split="train")
    prompts = [" ".join(review.split()[:4]) for review in imdb_train["text"]]

    trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=eval_prompts,
        config=config,
    )


if __name__ == "__main__":
    from datetime import date

    current_date = date.today()
    formatted_date = current_date.strftime("%Y-%m-%d")
    print(formatted_date)
    parser = argparse.ArgumentParser(description='Train a model with FPPO.')
    parser.add_argument('--json_config', default='', help='Type of divergence for FPPO. Default: reverse_kl')
    parser.add_argument('--f_divergence', default='reverse_kl', help='Type of divergence for FPPO. Default: reverse_kl')
    parser.add_argument('--alpha', default=0.5, type=float, help='Alpha for divergence calculation. Default: 0.5')
    parser.add_argument('--init_kl_coef', default=0.1, type=float, help='coefficient for  divergence loss. Default: 0.1')
    parser.add_argument('--kl_in_reward', type=bool, default=False, help='Whether to include kl divergence in reward. Default: False')
    args = parser.parse_args()
    if args.f_divergence == 'alpha_divergence':
        sys.argv[0] = f"{formatted_date}_{args.kl_in_reward}_{args.f_divergence}_alpha_{args.alpha}_{args.init_kl_coef}_{sys.argv[0]}"
    else:
        sys.argv[0] = f"{formatted_date}_{args.kl_in_reward}_{args.f_divergence}_{args.init_kl_coef}_{sys.argv[0]}"
    print(sys.argv[0])
    
    if len(args.json_config) > 1:
        hparams = json.loads(args.json_config)
    else:
        hparams = {}
    additional_hparams = vars(args)
    main(hparams, additional_hparams)


# # Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# # with a sentiment reward function
# import argparse
# import json
# import os
# import sys
# from typing import List

# import torch
# from datasets import load_dataset
# from transformers import pipeline

# import trlx
# from trlx.data.default_configs import TRLConfig
# from .configs import default_fppo_config


# def get_positive_score(scores):
#     "Extract value associated with a positive sentiment from pipeline's output"
#     return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


# def main(hparams={}):
#     # Merge sweep config with default config if given
#     config = TRLConfig.update(default_fppo_config(hparams.get('f_divergence', 'reverse_kl'), hparams.get('alpha', 0.5)).to_dict(), hparams)

#     if torch.cuda.is_available():
#         device = int(os.environ.get("LOCAL_RANK", 0))
#     else:
#         device = -1

#     sentiment_fn = pipeline(
#         "sentiment-analysis",
#         "lvwerra/distilbert-imdb",
#         top_k=2,
#         truncation=True,
#         batch_size=256,
#         device=device,
#     )

#     def reward_fn(samples: List[str], **kwargs) -> List[float]:
#         sentiments = list(map(get_positive_score, sentiment_fn(samples)))
#         return sentiments

#     # Take few words off of movies reviews as prompts
#     imdb = load_dataset("imdb", split="train+test")
#     prompts = [" ".join(review.split()[:4]) for review in imdb["text"]]

#     trlx.train(
#         reward_fn=reward_fn,
#         prompts=prompts,
#         eval_prompts=["I don't know much about Hungarian underground"] * 256,
#         config=config,
#     )


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Train a model with FPPO.')
#     parser.add_argument('--f_divergence', default='reverse_kl', help='Type of divergence for FPPO. Default: reverse_kl')
#     parser.add_argument('--alpha', default=0.5, type=float, help='Alpha for divergence calculation. Default: 0.5')
#     args = parser.parse_args()

#     hparams = vars(args)
#     main(hparams)
