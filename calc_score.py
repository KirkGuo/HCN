import os
import json

from nlgeval import compute_metrics, NLGEval
from tqdm import tqdm

import sys


def __calculate_scores(result_file, ref_file, block_print=True):
    reference_file = json.load(open(ref_file))
    ref_video_keys = sorted(list(reference_file.keys()))
    ref_text_list = sum([reference_file[item]['sentences'] for item in ref_video_keys], [])

    file_data = json.load(open(result_file))
    hyp_text_list = sum([[i['sentence'].lower() for i in file_data['results'][item]] for item in ref_video_keys], [])
    hyp_text_list = ['<NONE>' if len(item) == 0 else item for item in hyp_text_list]  # for empty generated result
    nlgeval = NLGEval(no_skipthoughts=True, no_glove=True)

    result = nlgeval.compute_metrics(hyp_list=ref_text_list, ref_list=[hyp_text_list])
    metrics = {
        'Average across tIoUs': result
    }
    return metrics


def main():
    # args = config.parse_config()

    # args.log_path = os.path.join(args.log_path, f'{timestamp}')
    # print(__calculate_scores( 'result_e10.json' , 'val.json'))
    bleu3 = -float('inf')
    bleu4 = -float('inf')
    meteor = -float('inf')
    rouge = -float('inf')
    cider = -float('inf')
    for num in tqdm(range(10, 30)):
        data = __calculate_scores(f'/home/leiji/Desktop/Kirk_G/distribute_model/log/1610427176/inferences/result_e{num}.json',
                             '/home/leiji/Desktop/Kirk_G/distribute_model/data/val.json')['Average across tIoUs']
        bleu3 = max(bleu3, data['Bleu_3'])
        bleu4 = max(bleu4, data['Bleu_4'])
        meteor = max(meteor, data['METEOR'])
        rouge = max(rouge, data['ROUGE_L'])
        cider = max(cider, data['CIDEr'])

    print(bleu3*100, bleu4*100, meteor*100, rouge*100, cider*100)

if __name__ == '__main__':
    main()