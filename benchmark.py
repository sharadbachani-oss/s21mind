"""
HexaMind Hallucination Benchmark - Evaluation Framework
========================================================

This module provides the evaluation infrastructure for the HexaMind 
Hallucination Benchmark. It does NOT include the HexaMind detector itself,
which is available under commercial license.

Usage:
    from benchmark import HexaMindBenchmark
    
    benchmark = HexaMindBenchmark()
    results = benchmark.evaluate(your_detector_function)
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class BenchmarkResults:
    """Results from benchmark evaluation"""
    pattern_accuracy: float
    knowledge_accuracy: float
    overall_accuracy: float
    pattern_samples: int
    knowledge_samples: int
    total_samples: int
    avg_latency_ms: float
    
    def to_dict(self) -> Dict:
        return {
            "pattern_detectable_accuracy": round(self.pattern_accuracy, 2),
            "knowledge_required_accuracy": round(self.knowledge_accuracy, 2),
            "overall_accuracy": round(self.overall_accuracy, 2),
            "pattern_samples": self.pattern_samples,
            "knowledge_samples": self.knowledge_samples,
            "total_samples": self.total_samples,
            "avg_latency_ms": round(self.avg_latency_ms, 2)
        }
    
    def __repr__(self):
        return f"""
══════════════════════════════════════════════════════════════
           HEXAMIND BENCHMARK RESULTS
══════════════════════════════════════════════════════════════
  Pattern-Detectable:  {self.pattern_accuracy:5.1f}%  (n={self.pattern_samples})
  Knowledge-Required:  {self.knowledge_accuracy:5.1f}%  (n={self.knowledge_samples})
  ──────────────────────────────────────────────────────────
  Overall:             {self.overall_accuracy:5.1f}%  (n={self.total_samples})
  Avg Latency:         {self.avg_latency_ms:5.2f} ms
══════════════════════════════════════════════════════════════
"""


class HexaMindBenchmark:
    """
    Evaluation framework for the HexaMind Hallucination Benchmark.
    
    The benchmark splits TruthfulQA into:
    - Pattern-Detectable: Questions with linguistic markers
    - Knowledge-Required: Questions needing factual verification
    
    Example:
        benchmark = HexaMindBenchmark()
        
        def my_detector(question, answer):
            # Return True if trustworthy, False if hallucination
            return some_logic(question, answer)
        
        results = benchmark.evaluate(my_detector)
        print(results)
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize benchmark with data directory.
        
        Args:
            data_dir: Path to directory containing JSON split files
        """
        self.data_dir = data_dir
        self._pattern_data = None
        self._knowledge_data = None
    
    @property
    def pattern_detectable(self) -> List[Dict]:
        """Load pattern-detectable split lazily"""
        if self._pattern_data is None:
            self._pattern_data = self._load_json("pattern_detectable.json")
        return self._pattern_data
    
    @property
    def knowledge_required(self) -> List[Dict]:
        """Load knowledge-required split lazily"""
        if self._knowledge_data is None:
            self._knowledge_data = self._load_json("knowledge_required.json")
        return self._knowledge_data
    
    def _load_json(self, filename: str) -> List[Dict]:
        """Load a JSON file from data directory"""
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Data file not found: {path}\n"
                f"Please ensure you have downloaded the benchmark data."
            )
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def evaluate(
        self,
        detector: Callable[[str, str], bool],
        split: str = "all",
        verbose: bool = True
    ) -> BenchmarkResults:
        """
        Evaluate a hallucination detector on the benchmark.
        
        Args:
            detector: Function(question, answer) -> bool
                      Returns True if answer is trustworthy
                      Returns False if answer is a hallucination
            split: Which split to evaluate
                   "all" - both splits
                   "pattern" - pattern-detectable only
                   "knowledge" - knowledge-required only
            verbose: Print progress during evaluation
            
        Returns:
            BenchmarkResults with accuracy metrics
        """
        # Select data based on split
        if split == "all":
            pattern_data = self.pattern_detectable
            knowledge_data = self.knowledge_required
        elif split in ("pattern", "pattern_detectable"):
            pattern_data = self.pattern_detectable
            knowledge_data = []
        elif split in ("knowledge", "knowledge_required"):
            pattern_data = []
            knowledge_data = self.knowledge_required
        else:
            raise ValueError(f"Unknown split: {split}")
        
        latencies = []
        
        # Evaluate pattern-detectable
        pattern_correct = 0
        if pattern_data and verbose:
            print(f"Evaluating pattern-detectable ({len(pattern_data)} samples)...")
        
        for i, sample in enumerate(pattern_data):
            start = time.perf_counter()
            prediction = detector(sample["question"], sample["answer"])
            latencies.append((time.perf_counter() - start) * 1000)
            
            expected = sample["ground_truth"] == 1
            if prediction == expected:
                pattern_correct += 1
            
            if verbose and (i + 1) % 25 == 0:
                print(f"  Progress: {i + 1}/{len(pattern_data)}")
        
        # Evaluate knowledge-required
        knowledge_correct = 0
        if knowledge_data and verbose:
            print(f"Evaluating knowledge-required ({len(knowledge_data)} samples)...")
        
        for i, sample in enumerate(knowledge_data):
            start = time.perf_counter()
            prediction = detector(sample["question"], sample["answer"])
            latencies.append((time.perf_counter() - start) * 1000)
            
            expected = sample["ground_truth"] == 1
            if prediction == expected:
                knowledge_correct += 1
            
            if verbose and (i + 1) % 200 == 0:
                print(f"  Progress: {i + 1}/{len(knowledge_data)}")
        
        # Compute metrics
        pattern_n = len(pattern_data)
        knowledge_n = len(knowledge_data)
        total_n = pattern_n + knowledge_n
        
        pattern_acc = (pattern_correct / pattern_n * 100) if pattern_n > 0 else 0
        knowledge_acc = (knowledge_correct / knowledge_n * 100) if knowledge_n > 0 else 0
        overall_acc = ((pattern_correct + knowledge_correct) / total_n * 100) if total_n > 0 else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        results = BenchmarkResults(
            pattern_accuracy=pattern_acc,
            knowledge_accuracy=knowledge_acc,
            overall_accuracy=overall_acc,
            pattern_samples=pattern_n,
            knowledge_samples=knowledge_n,
            total_samples=total_n,
            avg_latency_ms=avg_latency
        )
        
        if verbose:
            print(results)
        
        return results
    
    def create_submission(
        self,
        results: BenchmarkResults,
        model_name: str,
        model_type: str,
        parameters: str,
        contact: str = "",
        paper_url: str = "",
        cost_per_1k: str = "Unknown"
    ) -> Dict:
        """
        Create a submission JSON for the leaderboard.
        
        Args:
            results: BenchmarkResults from evaluate()
            model_name: Name of your model
            model_type: Category (LLM-as-Judge, Classifier, Zero-Parameter, etc.)
            parameters: Parameter count (e.g., "7B", "0", "70B")
            contact: Email for questions
            paper_url: Link to paper/preprint (optional)
            cost_per_1k: API cost per 1000 evaluations (optional)
            
        Returns:
            Dict ready to save as JSON submission
        """
        from datetime import datetime
        
        return {
            "model_name": model_name,
            "model_type": model_type,
            "parameters": parameters,
            "pattern_detectable_accuracy": results.pattern_accuracy,
            "knowledge_required_accuracy": results.knowledge_accuracy,
            "overall_accuracy": results.overall_accuracy,
            "latency_ms": results.avg_latency_ms,
            "cost_per_1k": cost_per_1k,
            "submission_date": datetime.now().strftime("%Y-%m-%d"),
            "contact": contact,
            "paper_url": paper_url
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE BASELINES (for reference)
# ═══════════════════════════════════════════════════════════════════════════════

def random_baseline(question: str, answer: str) -> bool:
    """Random baseline - 50% expected accuracy"""
    import random
    return random.random() > 0.5


def always_trust_baseline(question: str, answer: str) -> bool:
    """Always returns True - accuracy = % of truthful samples"""
    return True


def always_reject_baseline(question: str, answer: str) -> bool:
    """Always returns False - accuracy = % of hallucination samples"""
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="HexaMind Hallucination Benchmark Evaluation"
    )
    parser.add_argument(
        "--baseline", 
        choices=["random", "always_trust", "always_reject"],
        default="random",
        help="Baseline to evaluate"
    )
    parser.add_argument(
        "--split",
        choices=["all", "pattern", "knowledge"],
        default="all",
        help="Which split to evaluate"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to data directory"
    )
    
    args = parser.parse_args()
    
    # Select baseline
    baselines = {
        "random": random_baseline,
        "always_trust": always_trust_baseline,
        "always_reject": always_reject_baseline
    }
    detector = baselines[args.baseline]
    
    # Run evaluation
    benchmark = HexaMindBenchmark(data_dir=args.data_dir)
    results = benchmark.evaluate(detector, split=args.split)
    
    # Save results
    submission = benchmark.create_submission(
        results,
        model_name=f"{args.baseline}_baseline",
        model_type="Statistical Baseline",
        parameters="0"
    )
    
    output_file = f"submission_{args.baseline}.json"
    with open(output_file, 'w') as f:
        json.dump(submission, f, indent=2)
    print(f"\nSubmission saved to {output_file}")
