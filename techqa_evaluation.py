import collections
import json
import logging
import sys

import argparse
from typing import Dict, List, Tuple, Optional, Union

_NEGATIVE_INFINITY = float('-inf')
_DEFAULT_TOP_K = 5


class EVAL_OPTS():
    def __init__(self, data_file, pred_file, out_file="", top_k=5,
                 out_image_dir=None, verbose=False):
        self.data_file = data_file
        self.pred_file = pred_file
        self.out_file = out_file
        self.verbose = verbose
        self.top_k = top_k


OPTS = EVAL_OPTS(data_file=None, pred_file=None)

ScoresById = Dict[str, Union[int, float]]
TopKScoresById = Dict[str, List[Union[int, float]]]


def parse_args():
    parser = argparse.ArgumentParser(
        """
        Official evaluation script for TechQA v1. It will produce the following metrics: 
        
        - "QA_F1": Calculated for precision/recall based on character offset. The threshold 
        provided in the prediction json will be applied to predict NO ANSWER in cases 
        where the prediction score < threshold.
        
        - "IR_Precision": Calculated based on doc id match. The threshold provided in the 
        prediction json will be applied to predict NO ANSWER in cases where the prediction
         score < threshold.
        
        - "HasAns_QA_F1": Same as `QA_F1`, but calculated only on answerable questions. 
        Thresholds are ignored for this calculation.
        
        - "HasAns_Top_k_QA_F1": The max `QA_F1` based on the top `k` predictions calculated 
        only on answerable questions. Thresholds are ignored for this calculation. 
        By default k=%d.
        
        - "HasAns_IR_Precision": Same as `IR_Precision`, but calculated only on answerable 
        questions. Thresholds are ignored for this calculation.
        
        - "HasAns_Top_k_IR_Precision": The max `IR_Precision` based on the top `k` predictions
         calculated only on answerable questions. Thresholds are ignored for this calculation. 
        By default k=%d.
        
        - "Best_QA_F1": Same as `QA_F1`, but instead of applying the provided threshold, it 
        will scan for the `optimal` threshold based on the evaluation set.
        
        - "Best_QA_F1_Threshold": The threshold identified during the search for `Best_QA_F1`
        
        - "_Total_Questions": All metrics will be accompanied by a `_Total_Questions` count of
         the number of queries used to compute the statistic.
         """ % (_DEFAULT_TOP_K, _DEFAULT_TOP_K))
    parser.add_argument('data_file', metavar='dev_vX.json',
                        help='Input competition query annotations JSON file.')
    parser.add_argument('pred_file', metavar='pred.json',
                        help=
                        """
                        Model predictions JSON file in the format: 
                             {                              
                                "threshold": 0,              
                                "predictions": {             
                                  "QID1": [                  
                                    {                        
                                      "doc_id": "swg234",    
                                      "score": 3.4,          
                                      "start_offset": 0,     
                                      "end_offset": 100      
                                    },                       
                                    {                        
                                      "doc_id": "swg234",    
                                      "score": 3,            
                                      "start_offset": 50,    
                                      "end_offset": 100      
                                    }...                        
                                  ],                         
                                  "QID2": [                  
                                    {                        
                                      "doc_id": "",          
                                      "score": 0,            
                                      "start_offset": -1,    
                                      "end_offset": -1       
                                    },                       
                                    {                        
                                      "doc_id": "swg123",    
                                      "score": -1,           
                                      "start_offset": 20,    
                                      "end_offset": 30       
                                    }...                        
                                  ]...                          
                                }                            
                             }
                             """)
    parser.add_argument('--out-file', '-o', metavar='eval.json',
                        help='Write accuracy metrics to file (default is stdout).')
    parser.add_argument('--top_k', '-k', type=int, default=_DEFAULT_TOP_K,
                        help='Eval script will compute F1 score using the top 1 prediction'
                             ' as well as the top k predictions')
    parser.add_argument('--verbose', '-v', action="store_const", const=logging.DEBUG,
                        default=logging.INFO)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def make_qid_to_has_ans(dataset):
    qid_to_has_ans = {}
    for qid, q in dataset.items():
        if 'ANSWERABLE' in q and q['ANSWERABLE'] == 'Y':
            qid_to_has_ans[qid] = True
        else:
            qid_to_has_ans[qid] = False

    return qid_to_has_ans


def compute_f1(gold_start_offset, gold_end_offset, prediction_start_offset, prediction_end_offset):
    num_gold_chars = gold_end_offset - gold_start_offset
    num_pred_chars = prediction_end_offset - prediction_start_offset
    num_same_chars = max(0,
                         min(gold_end_offset, prediction_end_offset) - max(gold_start_offset,
                                                                           prediction_start_offset))
    if num_gold_chars == 0 or num_pred_chars == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(num_gold_chars == num_pred_chars)
    if num_same_chars == 0:
        return 0
    precision = 1.0 * num_same_chars / num_pred_chars
    recall = 1.0 * num_same_chars / num_gold_chars
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores(
        dataset: Dict[str, Dict], preds: Dict[str, List[Dict]],
        qid_to_has_ans: Dict[str, bool], top_k: int) -> Tuple[
    TopKScoresById, TopKScoresById, TopKScoresById]:
    prediction_scores_by_qid = {}
    f1_scores_by_qid = {}
    retrieval_accuracies_by_qid = {}

    for qid, q in dataset.items():

        prediction_scores = list()
        f1_scores = list()
        retrieval_accuracies = list()
        if qid not in preds or len(preds[qid]) < 1:
            logging.warning('Missing predictions for %s; going to receive 0 points for it' % qid)
            # Force this score to be incorrect
            prediction_scores.append(float('inf'))
            f1_scores.append(0)
            retrieval_accuracies.append(0)
        else:
            if qid_to_has_ans[qid]:
                gold_doc_id = q['DOCUMENT']
                gold_start_offset = int(q['START_OFFSET'])
                gold_end_offset = int(q['END_OFFSET'])
            else:
                gold_start_offset = -1
                gold_end_offset = -1
                gold_doc_id = ''

            for prediction in preds[qid][:top_k]:
                if gold_doc_id.strip() != prediction['doc_id'].strip():
                    f1_scores.append(0)
                    retrieval_accuracies.append(0)
                else:
                    f1_scores.append(compute_f1(gold_start_offset=gold_start_offset,
                                                gold_end_offset=gold_end_offset,
                                                prediction_start_offset=prediction['start_offset'],
                                                prediction_end_offset=prediction['end_offset']))
                    retrieval_accuracies.append(1)
                prediction_scores.append(prediction['score'])

        f1_scores_by_qid[qid] = f1_scores
        prediction_scores_by_qid[qid] = prediction_scores
        retrieval_accuracies_by_qid[qid] = retrieval_accuracies
    return f1_scores_by_qid, retrieval_accuracies_by_qid, prediction_scores_by_qid


def apply_no_ans_threshold(
        eval_scores: TopKScoresById, answer_probabilities: TopKScoresById,
        qid_to_has_ans: Dict[str, bool], answer_threshold: float) -> Tuple[ScoresById, ScoresById]:
    top1_eval_scores = {}
    max_eval_scores = {}

    for qid, s in eval_scores.items():
        # Check the top 1 prediction
        pred_na = answer_probabilities[qid][0] < answer_threshold
        if pred_na:
            top1_eval_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            top1_eval_scores[qid] = s[0]

        # Check all predictions
        if not qid_to_has_ans[qid] and any(
                score < answer_threshold for score in answer_probabilities[qid]):
            max_eval_scores[qid] = 1
        else:
            max_eval_scores[qid] = max(s)

    return top1_eval_scores, max_eval_scores


def make_eval_dict(f1_scores_by_qid: ScoresById, retrieval_scores_by_qid: ScoresById,
                   qid_list: Optional[set] = None) -> collections.OrderedDict:
    f1_score_sum = 0
    retrieval_score_sum = 0

    if not qid_list:
        qid_list = list(f1_scores_by_qid.keys())

    total = len(qid_list)
    for qid in qid_list:
        f1_score_sum += f1_scores_by_qid[qid]
        retrieval_score_sum += retrieval_scores_by_qid[qid]

    return collections.OrderedDict([
        ('QA_F1', 100.0 * f1_score_sum / total),
        ('IR_Precision', 100.0 * retrieval_score_sum / total),
        ('Total_Questions', total),
    ])


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval['%s_%s' % (prefix, k)] = new_eval[k]


def find_best_thresh(preds_by_qid, eval_scores_by_qid, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = float('inf')
    qid_list = sorted(preds_by_qid.keys(), key=lambda qid: preds_by_qid[qid], reverse=True)
    for i, qid in enumerate(qid_list):
        if qid not in eval_scores_by_qid: continue
        if qid_to_has_ans[qid]:
            diff = eval_scores_by_qid[qid]
        else:
            if preds_by_qid[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = preds_by_qid[qid]
    return 100.0 * best_score / len(eval_scores_by_qid), best_thresh


def find_all_best_thresh(main_eval, preds, f1_raw, qid_to_has_ans):
    best_f1, f1_thresh = find_best_thresh(preds, f1_raw, qid_to_has_ans)
    main_eval['Best_QA_F1'] = best_f1
    main_eval['Best_QA_F1_Threshold'] = f1_thresh


def main(OPTS):
    logging.basicConfig(level=OPTS.verbose)

    with open(OPTS.data_file, encoding='utf-8') as f:
        dataset = {query['QUESTION_ID']: query for query in json.load(f)}
    with open(OPTS.pred_file, encoding='utf-8') as f:
        system_output = json.load(f)
        threshold = system_output['threshold']
        preds = system_output['predictions']

    out_eval = evaluate(preds=preds, dataset=dataset, threshold=threshold)

    if OPTS.out_file:
        with open(OPTS.out_file, 'w') as f:
            json.dump(out_eval, f)
    else:
        print(json.dumps(out_eval, indent=2))
    return out_eval


def evaluate(preds: Dict[str, List[Dict]], dataset: Dict[str, Dict],
             threshold: float = _NEGATIVE_INFINITY):
    qid_to_has_ans = make_qid_to_has_ans(dataset)  # maps qid to True/False
    has_ans_qids = {k for k, v in qid_to_has_ans.items() if v}

    # Calculate metrics without thresholding
    f1_raw_by_qid, retrieval_acc_raw_by_qid, pred_score_by_qid = \
        get_raw_scores(dataset, preds, qid_to_has_ans, OPTS.top_k)
    top1_raw_retrieval_acc_by_qid = {qid: scores[0] for qid, scores in
                                     retrieval_acc_raw_by_qid.items()}
    topk_raw_retrieval_acc_by_qid = {qid: max(scores) for qid, scores in
                                     retrieval_acc_raw_by_qid.items()}
    top1_f1_raw_by_qid = {qid: scores[0] for qid, scores in f1_raw_by_qid.items()}
    topk_f1_raw_by_qid = {qid: max(scores) for qid, scores in f1_raw_by_qid.items()}
    top1_pred_score_by_qid = {qid: scores[0] for qid, scores in pred_score_by_qid.items()}

    # Calculating f1 with threshold
    top1_f1_thresh_by_qid, topk_f1_thresh_by_qid = apply_no_ans_threshold(f1_raw_by_qid,
                                                                          pred_score_by_qid,
                                                                          qid_to_has_ans, threshold)

    # Calculating doc retrieval accuracy with threshold
    top1_retrieval_acc_by_qid, topk_retrieval_acc_by_qid = \
        apply_no_ans_threshold(retrieval_acc_raw_by_qid, pred_score_by_qid,
                               qid_to_has_ans, threshold)

    # Create evaluation summary
    out_eval = make_eval_dict(top1_f1_thresh_by_qid, top1_retrieval_acc_by_qid)
    if has_ans_qids:
        merge_eval(out_eval, make_eval_dict(top1_f1_raw_by_qid, top1_raw_retrieval_acc_by_qid,
                                            qid_list=has_ans_qids), 'HasAns')
        merge_eval(out_eval, make_eval_dict(topk_f1_raw_by_qid, topk_raw_retrieval_acc_by_qid,
                                            qid_list=has_ans_qids),
                   'HasAns_Top_%d' % OPTS.top_k)

    # Find best threshold for top 1 f1 metric
    find_all_best_thresh(out_eval, top1_pred_score_by_qid, top1_f1_raw_by_qid, qid_to_has_ans)

    return out_eval


if __name__ == '__main__':
    OPTS = parse_args()
    main(OPTS)