
Run the following to get calibration experiments for hh-dataset

```
python hh_cali_eval.py --checkpoint /path/to/fined-tuned-model/checkpoint --ref_checkpoint /path/to/reference-model/checkpoint --rootdir /where/results/are/stored --data_path /path/to/data`
```


Run the following scripts to get ECE numbers and plots.

```
python cali_plot.py --rootdir /where/results/are/stored --div_name divergence-name --alpha alpha-for-alpha-divergence
```