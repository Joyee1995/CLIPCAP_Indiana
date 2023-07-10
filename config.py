import os
from sacred import Experiment
ex = Experiment("MedicalReportGeneration")
base_dir = os.path.dirname(__file__)

@ex.config
def config():
    # basic
    name = "MedicalClipCap" # "<impression/finding>_<clip/mimic>_<gpt2/medgpt>_<prefix/gpt>"
    seed = 2023
    device = 'cuda' # cuda
    version = None
    wandb_api_key_fp = os.path.join(base_dir, "wandb_api_key.txt") # https://wandb.ai/authorize
        
    # global directory
    indiana_dir = "./data/chest-xrays-indiana-university"
    log_dir = os.path.abspath(os.path.join(base_dir, "logs"))
    model_dir = os.path.abspath(os.path.join(base_dir, "ckpts"))
    preprocess_dir = os.path.abspath(os.path.join(base_dir, "preprocess"))
    medical_clip = True

    # pretrained_weights (for parse_indiana.py)
    clip_fp = "./pretrain_weights/ViT-B-32.pt"
    medical_clip_fp = "./pretrain_weights/clip-imp-pretrained_128_6_after_4.pt"
    data_json_src = ["preprocess/impression_train.json", "preprocess/impression_valid.json"]

    # Dataset
    num_workers = 12
    train_datapath = os.path.join(base_dir, "preprocess", "impression_train_clip_embed.pkl")
    valid_datapath = os.path.join(base_dir, "preprocess", "impression_valid_clip_embed.pkl")

    # for training
    save_top_k = 1
    bs = 8
    lr = 2e-5
    epochs = 50
    num_layers = 8
    warmup_steps = 5000
    prefix_length = 40
    prefix_length_clip = 40

    # For Model
    ckpt_path = None
    gpt2_type = "stanford-crfm/BioMedLM" # "gpt2"
    is_rn = False # check
    only_prefix = False # True
    normalize_prefix = False
    mapping_type = 'transformer'
    functional_test_size = None