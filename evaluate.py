import re
import string
from collections import Counter
import json

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth_list):
    normalized_prediction = normalize_answer(prediction)
    
    max_f1 = 0
    max_precision = 0
    max_recall = 0
    
    for ground_truth in ground_truth_list:
        normalized_ground_truth = normalize_answer(ground_truth)
        
        ZERO_METRIC = (0, 0, 0)
        
        if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
            continue
        if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
            continue
            
        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            continue
            
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        if f1 > max_f1:
            max_f1 = f1
            max_precision = precision
            max_recall = recall
    
    return max_f1, max_precision, max_recall

def exact_match_score(prediction, ground_truth_list):
    normalized_prediction = normalize_answer(prediction)
    return any(normalize_answer(truth) == normalized_prediction for truth in ground_truth_list)

def evaluate_answers_subq(answers_file, output_file):
    metrics_with_subq = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}
    metrics_without_subq = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}
    results = []
    
    try:
        with open(answers_file, 'r', encoding='utf-8') as f:
            content = f.read()
            content = "[" + content.replace("}\n{", "},{") + "]"
            answers = json.loads(content)
        
        total_count = len(answers)
        
        for i, answer in enumerate(answers):
            try:
                prediction_with_subq = answer['answer_with_subquestions'].strip('.')
                prediction_without_subq = answer['answer_without_subquestions'].strip('.')
                gold_list = answer['expected_answer']
                
                em_with_subq = exact_match_score(prediction_with_subq, gold_list)
                f1_with_subq, prec_with_subq, recall_with_subq = f1_score(prediction_with_subq, gold_list)
                
                em_without_subq = exact_match_score(prediction_without_subq, gold_list)
                f1_without_subq, prec_without_subq, recall_without_subq = f1_score(prediction_without_subq, gold_list)
                
                
                metrics_with_subq['em'] += float(em_with_subq)
                metrics_with_subq['f1'] += f1_with_subq
                metrics_with_subq['prec'] += prec_with_subq
                metrics_with_subq['recall'] += recall_with_subq
                
                metrics_without_subq['em'] += float(em_without_subq)
                metrics_without_subq['f1'] += f1_without_subq
                metrics_without_subq['prec'] += prec_without_subq
                metrics_without_subq['recall'] += recall_without_subq
                

                results.append({
                    'id': i + 1,
                    'question': answer['question'],
                    'gold_answers': gold_list,
                    'with_subquestions': {
                        'prediction': prediction_with_subq,
                        'exact_match': em_with_subq,
                        'f1_score': f1_with_subq,
                        'precision': prec_with_subq,
                        'recall': recall_with_subq
                    },
                    'without_subquestions': {
                        'prediction': prediction_without_subq,
                        'exact_match': em_without_subq,
                        'f1_score': f1_without_subq,
                        'precision': prec_without_subq,
                        'recall': recall_without_subq
                    }
                })
                
            except Exception as e:
                print(f"An error occurred while processing the {i+1}th answer: {str(e)}")
                print(f"Question answer: {answer}")
                continue
        
        processed_count = len(results)
        if processed_count > 0:
            for metrics in [metrics_with_subq, metrics_without_subq]:
                for k in metrics.keys():
                    metrics[k] = metrics[k] / processed_count
        
        output_data = {
            'total_questions': total_count,
            'processed_questions': processed_count,
            'metrics_with_subquestions': metrics_with_subq,
            'metrics_without_subquestions': metrics_without_subq,
            'results': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        
    except Exception as e:
        print(f"error in evaluate_answers_subq: {str(e)}")
        import traceback
        traceback.print_exc()
if __name__ == '__main__':
    evaluate_answers_subq("your_output.json", "your_output_eval.json")

