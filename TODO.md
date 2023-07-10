# Experiment: `<impression/finding>_<clip/mimic>_<gpt2/medgpt>_<prefix/gpt>`
1. [ ] impression_clip_gpt2_prefix
2. [ ] impression_mimic_medgpt_prefix
3. [ ] impression_mimic_medgpt_gpt
4. [ ] impression_clip_gpt2_gpt

# TODO
- add output dir in predict.py
- add preview output functions in predict.py
- Test Predict.py for torch lightning
- Try to find a way to sync ckpts to aws s3
- remove main ipynb

# DONE
- add wandb in training script
- move training log to logs directory
- Monitor Loss to save checkpoints
- sperate log dir into experiments
- move checkpoints to s3
- add key wandb key inside, and in the .gitignore
- Change the valid to 10%
- Decide to filter out the data with both frontal/lateral & finding/impression: follow the IU paper
- add seed everything

# suspend
