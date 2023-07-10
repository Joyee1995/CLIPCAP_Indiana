import json
import pickle
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def main(GTSpkl_fp, RESpkl_fp):
    GTSjson_fp = GTSpkl_fp.replace(".pkl", ".json")
    RESjson_fp = RESpkl_fp.replace(".pkl", ".json")
    with open(GTSpkl_fp, 'rb') as f:
        datasetGTS = pickle.load(f)        
        with open(GTSjson_fp, 'w') as f:
            json.dump(datasetGTS, f, ensure_ascii=True)

    # with open("datasetRES.json", 'w') as f:
    #     json.dump(datasetGTS['annotations'], f, ensure_ascii=False)

    with open(RESpkl_fp, 'rb') as f:
        datasetRES = pickle.load(f)
        with open(RESjson_fp, 'w') as f:
            json.dump(datasetRES['annotations'], f, ensure_ascii=False)
        
    annotation_file = GTSjson_fp
    results_file = RESjson_fp

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)    

    # evaluate on a subset of images by setting
    coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gts', required=True, type=str)
    parser.add_argument('--res', required=True, type=str)
    args = parser.parse_args()
    main(args.gts, args.res)


    