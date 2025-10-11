#!/usr/bin/env python3
"""
CoNLL-U evaluation script for dependency parsing.
Computes UAS, LAS, and other standard UD metrics.
Compatible with official CoNLL 2018 shared task evaluation.
"""
import argparse
from collections import defaultdict
from conllu import parse_incr

class EvaluationMetrics:
    """Accumulate and compute parsing metrics."""
    def __init__(self):
        self.total_tokens = 0
        self.correct_heads = 0
        self.correct_heads_labels = 0
        self.correct_trees = 0
        self.total_sentences = 0
        
        # Per-relation statistics
        self.per_label = defaultdict(lambda: {'total': 0, 'correct_uas': 0, 'correct_las': 0})
        
    def add(self, gold_sent, pred_sent):
        """Add evaluation for one sentence."""
        self.total_sentences += 1
        
        # Check sentence lengths match
        if len(gold_sent) != len(pred_sent):
            print(f"Warning: Sentence length mismatch ({len(gold_sent)} vs {len(pred_sent)})")
            return
        
        sent_correct_heads = True
        sent_correct_labels = True
        
        for gold_token, pred_token in zip(gold_sent, pred_sent):
            # Skip multiword tokens and empty nodes
            if not isinstance(gold_token['id'], int):
                continue
            if not isinstance(pred_token['id'], int):
                continue
                
            self.total_tokens += 1
            
            gold_head = gold_token['head']
            pred_head = pred_token['head']
            gold_label = gold_token['deprel']
            pred_label = pred_token['deprel']
            
            # UAS: correct head
            if gold_head == pred_head:
                self.correct_heads += 1
                self.per_label[gold_label]['correct_uas'] += 1
            else:
                sent_correct_heads = False
            
            # LAS: correct head and label
            if gold_head == pred_head and gold_label == pred_label:
                self.correct_heads_labels += 1
                self.per_label[gold_label]['correct_las'] += 1
            else:
                sent_correct_labels = False
            
            self.per_label[gold_label]['total'] += 1
        
        # Complete tree correct
        if sent_correct_heads and sent_correct_labels:
            self.correct_trees += 1
    
    def get_uas(self):
        """Unlabeled Attachment Score."""
        return self.correct_heads / max(1, self.total_tokens)
    
    def get_las(self):
        """Labeled Attachment Score."""
        return self.correct_heads_labels / max(1, self.total_tokens)
    
    def get_tree_accuracy(self):
        """Percentage of completely correct trees."""
        return self.correct_trees / max(1, self.total_sentences)
    
    def get_per_label_metrics(self):
        """Return per-label UAS and LAS."""
        results = {}
        for label, stats in self.per_label.items():
            if stats['total'] > 0:
                results[label] = {
                    'count': stats['total'],
                    'uas': stats['correct_uas'] / stats['total'],
                    'las': stats['correct_las'] / stats['total']
                }
        return results
    
    def print_summary(self):
        """Print evaluation summary."""
        print(f"\n{'='*70}")
        print("EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(f"Sentences: {self.total_sentences}")
        print(f"Tokens:    {self.total_tokens}")
        print(f"\nOverall Metrics:")
        print(f"  UAS (Unlabeled Attachment Score): {self.get_uas():.4f}")
        print(f"  LAS (Labeled Attachment Score):   {self.get_las():.4f}")
        print(f"  Tree Accuracy:                     {self.get_tree_accuracy():.4f}")
        
        # Per-label breakdown
        per_label = self.get_per_label_metrics()
        if per_label:
            print(f"\nPer-Relation Metrics:")
            print(f"{'Relation':<15} {'Count':>8} {'UAS':>8} {'LAS':>8}")
            print(f"{'-'*44}")
            for label in sorted(per_label.keys(), key=lambda x: per_label[x]['count'], reverse=True):
                stats = per_label[label]
                print(f"{label:<15} {stats['count']:>8} {stats['uas']:>8.4f} {stats['las']:>8.4f}")
        
        print(f"{'='*70}\n")

def read_conllu_file(filepath):
    """Read a CoNLL-U file and return list of sentences."""
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for sent in parse_incr(f):
            sentences.append(sent)
    return sentences

def align_sentences(gold_sents, pred_sents):
    """Align gold and predicted sentences."""
    if len(gold_sents) != len(pred_sents):
        print(f"Warning: Different number of sentences: gold={len(gold_sents)}, pred={len(pred_sents)}")
        min_len = min(len(gold_sents), len(pred_sents))
        return gold_sents[:min_len], pred_sents[:min_len]
    return gold_sents, pred_sents

def evaluate(gold_file, pred_file, verbose=False):
    """Evaluate predictions against gold standard."""
    print(f"Loading gold standard: {gold_file}")
    gold_sents = read_conllu_file(gold_file)
    print(f"✓ Loaded {len(gold_sents)} sentences")
    
    print(f"Loading predictions: {pred_file}")
    pred_sents = read_conllu_file(pred_file)
    print(f"✓ Loaded {len(pred_sents)} sentences")
    
    # Align sentences
    gold_sents, pred_sents = align_sentences(gold_sents, pred_sents)
    
    # Evaluate
    metrics = EvaluationMetrics()
    for i, (gold_sent, pred_sent) in enumerate(zip(gold_sents, pred_sents)):
        metrics.add(gold_sent, pred_sent)
        
        if verbose and i < 5:  # Show first few sentences
            print(f"\nSentence {i+1}:")
            print(f"  Gold heads: {[tok['head'] for tok in gold_sent if isinstance(tok['id'], int)]}")
            print(f"  Pred heads: {[tok['head'] for tok in pred_sent if isinstance(tok['id'], int)]}")
    
    # Print summary
    metrics.print_summary()
    
    return metrics

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate dependency parsing predictions in CoNLL-U format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate predictions
  python conll_eval.py gold.conllu pred.conllu
  
  # Verbose mode (show details for first sentences)
  python conll_eval.py gold.conllu pred.conllu --verbose
  
  # Save results to file
  python conll_eval.py gold.conllu pred.conllu > results.txt
        """
    )
    parser.add_argument("gold_file", type=str, help="Gold standard CoNLL-U file")
    parser.add_argument("pred_file", type=str, help="Predicted CoNLL-U file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    try:
        metrics = evaluate(args.gold_file, args.pred_file, args.verbose)
        
        # Return non-zero exit code if accuracy is suspiciously low
        if metrics.get_uas() < 0.1:
            print("⚠ Warning: UAS < 0.1, predictions may be incorrect format")
            return 1
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())

