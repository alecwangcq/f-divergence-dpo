export OPENAI_API_KEY=[YOUR_API]
echo "Now comparing rkl vs jsd in dpo"
#python gen_judgment.py --mode pairwise-baseline --model-list dpo_reverse_kl_pythia28_beta0_1 --baseline-model dpo_jsd_pythia28_beta0_1 --parallel 2
python show_result.py --mode pairwise-baseline --model-list dpo_reverse_kl_pythia28_beta0_1 --baseline-model dpo_jsd_pythia28_beta0_1
echo "Now comparing jsd vs alpha-div in dpo"
#python gen_judgment.py --mode pairwise-baseline --model-list dpo_jsd_pythia28_beta0_1 --baseline-model dpo_alpha_divergence_beta0_1_alpha0_5 --parallel 2
python show_result.py --mode pairwise-baseline --model-list dpo_jsd_pythia28_beta0_1 --baseline-model dpo_alpha_divergence_beta0_1_alpha0_5
echo "Now comparing alpha-div vs fkl in dpo"
#python gen_judgment.py --mode pairwise-baseline --model-list dpo_alpha_divergence_beta0_1_alpha0_5 --baseline-model dpo_forward_kl_pythia28_beta0_1 --parallel 2
python show_result.py --mode pairwise-baseline --model-list dpo_alpha_divergence_beta0_1_alpha0_5 --baseline-model dpo_forward_kl_pythia28_beta0_1

#for key in forward_kl reverse_kl jsd
##for key in forward_kl jsd
##for key in alpha_divergence_beta0_1_alpha0_5
#do
    #echo ${key}
    ##python gen_judgment.py --mode pairwise-baseline --model-list dpo_${key}_pythia28_beta0_1 --baseline-model  ppo_hh_${key}_0_1 --parallel 2
    #python show_result.py --mode pairwise-baseline --model-list dpo_${key}_pythia28_beta0_1 --baseline-model  ppo_hh_${key}_0_1
    ##python gen_judgment.py --mode pairwise-baseline --model-list dpo_${key} --baseline-model ppo_hh_alpha_0_5_beta_0_1 --parallel 2
    ##python show_result.py --mode pairwise-baseline --model-list dpo_${key} --baseline-model ppo_hh_alpha_0_5_beta_0_1
#done
##for key in forward_kl reverse_kl jsd
##for key in forward_kl jsd
#for key in alpha_divergence_beta0_1_alpha0_5
#do
    ##python gen_judgment.py --mode pairwise-baseline --model-list dpo_${key}_pythia28_beta0_1 --baseline-model  ppo_hh_${key}_0_1 --parallel 2
    ##python show_result.py --mode pairwise-baseline --model-list dpo_${key}_pythia28_beta0_1 --baseline-model  ppo_hh_${key}_0_1
    ##python gen_judgment.py --mode pairwise-baseline --model-list dpo_${key} --baseline-model ppo_hh_alpha_0_5_beta_0_1 --parallel 2
    #echo ${key}
    #python show_result.py --mode pairwise-baseline --model-list dpo_${key} --baseline-model ppo_hh_alpha_0_5_beta_0_1
#done
