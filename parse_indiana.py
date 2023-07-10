import json
import clip
import copy
import torch
import pickle
from tqdm import tqdm
from PIL import Image
import skimage.io as io
from config import ex, config

def encode_image_to_pkl(json_fp_src, pkl_fp_dst, clip_model, preprocess, device):
    assert json_fp_src.endswith(".json"), "json_fp should endswith .json"
    with open(json_fp_src, 'r') as f:
        data = json.load(f)
        print("%0d captions loaded from json " % len(data))

    ## continue process
    all_embeddings = []
    all_captions = []
    # if not os.path.exists(pkl_fp_dst):
    #     all_embeddings = []
    #     all_captions = []
    # else:
    #     with open(pkl_fp_dst, 'rb') as f:
    #         # {"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, 
    #         parsed_embedding = pickle.load(f)
    #         clip_embedding = parsed_embedding["clip_embedding"]
    #         all_embeddings = [clip_embedding[i:i+1] for i in range(clip_embedding.shape[0])]
    #         all_captions = parsed_embedding["captions"]
    # print("all_embeddings", len(all_embeddings))
    # print("all_captions", len(all_captions))
    
    # start process
    for i in tqdm(range(len(all_embeddings), len(data))):
        d = data[i]
        img_id = d["image_id"]
        filename = d['img_fp']
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        d["clip_embedding"] = i
        all_embeddings.append(prefix)
        all_captions.append(d)
        # if (i + 1) % 1000 == 0:
        #     with open(pkl_fp_dst, 'wb') as f:
        #         pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(pkl_fp_dst, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    print("pkl file saved to ", pkl_fp_dst)


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    device = torch.device(_config['device'])
    
    # model loading
    clip_model, preprocess = clip.load(_config['clip_fp'], device=device, jit=False)
    if _config['medical_clip']:
        clip_model.load_state_dict(torch.load(_config['medical_clip_fp'], map_location=device))
    clip_model = clip_model.to(device)

    # processing
    for json_fp_src in _config['data_json_src']:
        pkl_fp_dst = json_fp_src.replace(".json", "_clip_embed.pkl")
        encode_image_to_pkl(json_fp_src, pkl_fp_dst, clip_model, preprocess, device)
