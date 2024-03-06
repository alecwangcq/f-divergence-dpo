import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import itertools
import os
import copy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_bins', type=int, default=10)
    parser.add_argument('--rootdir', type=str, default='./hh')

    # chosen from 'forward_kl', 'jsd', 'reverse_kl', 'alpha_divergence'
    parser.add_argument('--div_name', type=str, default='alpha_divergence')
    parser.add_argument('--alpha', type=float, default=0.7)

    args = parser.parse_args()

    methods = [args.div_name]
    alpha = args.alpha
    betas = ['0.1', '0.3', '0.9']
    steps = [ "step-19968",  "step-39936",  "step-59904",  "step-79872",  "step-99840",  "step-119808", "step-139776",  "step-159744",  "LATEST"]

    ece_dict = {}
    for method, beta, step in itertools.product(*[methods, betas, steps]):
        print(method, beta, step)
        
        if method == 'alpha_divergence':
            model_name = 'dpo_' + method + '_beta' + beta
            model_name += '_alpha' + str(alpha)
        else:
            model_name = 'dpo_' + method + '_pythia28_beta' + beta

        test_path = os.path.join(args.rootdir, model_name, step)

        jsonl_file_path = test_path + '/test.jsonl' 
        df = pd.read_json(jsonl_file_path, lines=True)

        df['answers'] = [True] * len(df)

        df['prob'] = np.exp(df['chosen_score'] / df['chosen_length'])

        df['pred'] = (df['chosen_score'] / df['chosen_length']) > (df['reject_score'] / df['reject_length'])

        # calculate ECE
        bin_boundaries = np.linspace(0, 1, args.num_bins + 1)
        bin_indices = np.digitize(df['prob'], bin_boundaries)

        ece = 0.0
        total_samples = len(df)

        avg_confidence_list = []
        avg_accuracy_list = []
        for bin_idx in range(1, args.num_bins + 1):
            bin_mask = bin_indices == bin_idx
            if np.any(bin_mask):
                bin_confidences = df['prob'][bin_mask]
                bin_predicted_labels = df["pred"][bin_mask]
                bin_true_labels = df["answers"][bin_mask]

                accuracy_in_bin = np.mean(bin_predicted_labels == bin_true_labels)

                avg_confidence = np.mean(bin_confidences)

                avg_confidence_list.append(avg_confidence)
                avg_accuracy_list.append(accuracy_in_bin)

                ece += np.abs(accuracy_in_bin - avg_confidence) * len(bin_confidences)
            else:
                avg_confidence_list.append(0)
                avg_accuracy_list.append(0)

        ece /= total_samples

        print("ece: " + str(ece))

        if beta not in ece_dict:
            ece_dict[beta] = [ece]
        else:
            ece_dict[beta].append(ece)
            
    plt.figure()
    plt.plot(steps, ece_dict['0.1'], label='0.1')
    plt.plot(steps, ece_dict['0.3'], label='0.3')
    plt.plot(steps, ece_dict['0.9'], label='0.9')
    plt.legend()
    plt.title(method)
    plt.xticks(steps, steps, rotation=45)
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    if method == 'alpha_divergence':
        plt.savefig(method + '_alpha' + str(alpha) + '.png')
    else:
        plt.savefig(method + '.png')