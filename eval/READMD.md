```
# Evaluate on 0_random_140 with first 10 samples
python tools/evaluate_piebench.py --dataset_config 0_random_140 --max_samples 10

# Full evaluation on change_object_80
python tools/evaluate_piebench.py --dataset_config 1_change_object_80

# Custom output
python tools/evaluate_piebench.py --dataset_config 5_change_attribute_pose_40 \
  --output_dir eval_pose_outputs --csv_path eval_pose_outputs/results.csv
  ```




python eval/evaluate_piebench.py --dataset_config 0_random_140 --max_samples 5


python eval/evaluate_piebench.py --dataset_config 0_random_140 --max_samples 5 --dtype fp32


https://huggingface.co/datasets/UB-CVML-Group/PIE_Bench_pp/viewer?views%5B%5D=_0_random_140