#!/usr/bin/env python3
"""
MLIR Matmul Optimization Automation Script
This script serves as a template for Claude Code to autonomously optimize matmul operations.
"""

import os
import re
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

@dataclass
class BenchmarkResult:
    """Store benchmark results from submit command"""
    job_id: str
    base_time_ns: int
    optimized_time_ns: int
    pytorch_time_ns: int
    speedup_vs_base: float
    slowdown_vs_pytorch: float
    
    def __str__(self):
        return (f"Job {self.job_id}: "
                f"Speedup vs Base: {self.speedup_vs_base:.2f}x, "
                f"Slowdown vs PyTorch: {self.slowdown_vs_pytorch:.2f}x")

class MatmulOptimizer:
    """Autonomous MLIR Matmul Optimizer"""
    
    def __init__(self, project_dir: str = "."):
        self.project_dir = Path(project_dir)
        self.schedules_dir = self.project_dir / "schedules"
        self.logs_dir = self.project_dir / "logs"
        self.out_dir = self.project_dir / "out"
        self.iteration_counter = {}  # Track iterations per matmul
        self.results_log = []  # Store all results
        
        # Ensure directories exist
        self.schedules_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.out_dir.mkdir(exist_ok=True)
        
    def discover_matmuls(self) -> List[int]:
        """Find all matmul_*.mlir files and extract indices"""
        matmul_files = list(self.project_dir.glob("matmul_*.mlir"))
        indices = []
        for f in matmul_files:
            match = re.search(r'matmul_(\d+)\.mlir', f.name)
            if match:
                indices.append(int(match.group(1)))
        return sorted(indices)
    
    def get_next_iteration(self, matmul_index: int) -> int:
        """Get the next claude iteration number for a matmul"""
        if matmul_index not in self.iteration_counter:
            # Check existing claude schedules
            existing = list(self.schedules_dir.glob(f"claude*_{matmul_index}.mlir"))
            if existing:
                max_iter = max(
                    int(re.search(r'claude(\d+)_', f.name).group(1))
                    for f in existing
                )
                self.iteration_counter[matmul_index] = max_iter + 1
            else:
                self.iteration_counter[matmul_index] = 1
        else:
            self.iteration_counter[matmul_index] += 1
        
        return self.iteration_counter[matmul_index]
    
    def submit_benchmark(self, matmul_index: int, schedule_name: str, 
                        no_bufferize: bool = False) -> Optional[BenchmarkResult]:
        """Run submit command and parse results"""
        cmd = ["./submit", f"-{matmul_index}"]
        if no_bufferize:
            cmd.append("-no-bufferize")
        cmd.append(schedule_name)
        
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                cwd=self.project_dir
            )
            
            output = result.stdout
            print(output)
            
            # Parse the output
            return self._parse_benchmark_output(output)
            
        except subprocess.TimeoutExpired:
            print(f"ERROR: Benchmark timed out for {schedule_name}")
            return None
        except Exception as e:
            print(f"ERROR: Failed to run benchmark: {e}")
            return None
    
    def _parse_benchmark_output(self, output: str) -> Optional[BenchmarkResult]:
        """Parse benchmark output to extract metrics"""
        try:
            job_id_match = re.search(r'Job (\S+) submitted', output)
            base_match = re.search(r'Base:\s*Execution time \(ns\):\s*(\d+)', output)
            opt_match = re.search(r'Optimized:\s*Execution time \(ns\):\s*(\d+)', output)
            pytorch_match = re.search(r'PyTorch:\s*Execution time \(ns\):\s*(\d+)', output)
            speedup_match = re.search(r'Speedup over Base:\s*([\d.]+)x', output)
            slowdown_match = re.search(r'Slowdown compared to PyTorch:\s*([\d.]+)x', output)
            
            if all([job_id_match, base_match, opt_match, pytorch_match, 
                   speedup_match, slowdown_match]):
                return BenchmarkResult(
                    job_id=job_id_match.group(1),
                    base_time_ns=int(base_match.group(1)),
                    optimized_time_ns=int(opt_match.group(1)),
                    pytorch_time_ns=int(pytorch_match.group(1)),
                    speedup_vs_base=float(speedup_match.group(1)),
                    slowdown_vs_pytorch=float(slowdown_match.group(1))
                )
        except Exception as e:
            print(f"ERROR: Failed to parse benchmark output: {e}")
        
        return None
    
    def run_lower(self, matmul_index: int, schedule_name: str,
                  no_bufferize: bool = False) -> bool:
        """Run lower command to generate intermediate representations"""
        # Clean output directory
        print("Cleaning out/ directory...")
        for f in self.out_dir.glob("*"):
            f.unlink()
        
        cmd = ["./lower", f"-{matmul_index}"]
        if no_bufferize:
            cmd.append("-no-bufferize")
        cmd.append(schedule_name)
        
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=self.project_dir
            )
            
            print(result.stdout)
            if result.returncode != 0:
                print(f"ERROR: Lower command failed: {result.stderr}")
                return False
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to run lower: {e}")
            return False
    
    def analyze_lowered_output(self) -> Dict[str, str]:
        """Read and return lowered output files for analysis"""
        outputs = {}
        for filename in ["output.mlir", "output.ll.mlir", "output.ll", "output.s"]:
            filepath = self.out_dir / filename
            if filepath.exists():
                outputs[filename] = filepath.read_text()
        return outputs
    
    def create_schedule(self, matmul_index: int, iteration: int,
                       schedule_content: str, passes_content: str) -> str:
        """Create schedule and passes files"""
        schedule_name = f"claude{iteration}_{matmul_index}"
        
        schedule_file = self.schedules_dir / f"{schedule_name}.mlir"
        passes_file = self.schedules_dir / f"{schedule_name}.txt"
        
        schedule_file.write_text(schedule_content)
        passes_file.write_text(passes_content)
        
        print(f"Created schedule: {schedule_name}")
        return schedule_name
    
    def log_result(self, matmul_index: int, iteration: int, 
                   result: BenchmarkResult, strategy: str):
        """Log optimization result"""
        entry = {
            "matmul_index": matmul_index,
            "iteration": iteration,
            "schedule_name": f"claude{iteration}_{matmul_index}",
            "job_id": result.job_id,
            "slowdown_vs_pytorch": result.slowdown_vs_pytorch,
            "speedup_vs_base": result.speedup_vs_base,
            "optimized_time_ns": result.optimized_time_ns,
            "pytorch_time_ns": result.pytorch_time_ns,
            "strategy": strategy,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.results_log.append(entry)
        
        # Also write to a log file
        log_file = self.project_dir / "optimization_log.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        
        print(f"\n{'='*60}")
        print(f"Matmul {matmul_index} - Iteration {iteration}")
        print(f"Strategy: {strategy}")
        print(f"Slowdown vs PyTorch: {result.slowdown_vs_pytorch:.4f}x")
        print(f"Speedup vs Base: {result.speedup_vs_base:.2f}x")
        print(f"{'='*60}\n")
    
    def get_best_result(self, matmul_index: int) -> Optional[Dict]:
        """Get best result so far for a matmul"""
        matmul_results = [r for r in self.results_log 
                         if r["matmul_index"] == matmul_index]
        if not matmul_results:
            return None
        
        return min(matmul_results, key=lambda x: x["slowdown_vs_pytorch"])
    
    def generate_report(self):
        """Generate final optimization report"""
        report = ["# MLIR Matmul Optimization Report\n"]
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        matmul_indices = sorted(set(r["matmul_index"] for r in self.results_log))
        
        for idx in matmul_indices:
            results = [r for r in self.results_log if r["matmul_index"] == idx]
            best = min(results, key=lambda x: x["slowdown_vs_pytorch"])
            
            report.append(f"## Matmul {idx}\n\n")
            report.append(f"**Best Result:** {best['schedule_name']}\n")
            report.append(f"- Slowdown vs PyTorch: **{best['slowdown_vs_pytorch']:.4f}x**\n")
            report.append(f"- Speedup vs Base: {best['speedup_vs_base']:.2f}x\n")
            report.append(f"- Strategy: {best['strategy']}\n\n")
            
            report.append("### Iteration History\n\n")
            report.append("| Iteration | Schedule | Slowdown vs PyTorch | Speedup vs Base | Strategy |\n")
            report.append("|-----------|----------|---------------------|-----------------|----------|\n")
            
            for r in results:
                report.append(
                    f"| {r['iteration']} | {r['schedule_name']} | "
                    f"{r['slowdown_vs_pytorch']:.4f}x | "
                    f"{r['speedup_vs_base']:.2f}x | "
                    f"{r['strategy'][:50]} |\n"
                )
            
            report.append("\n")
        
        report_path = self.project_dir / "OPTIMIZATION_REPORT.md"
        report_path.write_text("".join(report))
        print(f"\nReport generated: {report_path}")

# Helper functions for Claude Code to use with web search

def search_mlir_docs(query: str) -> str:
    """
    Template function for searching MLIR documentation.
    Claude Code should use web_search tool to implement this.
    
    Example searches:
    - "MLIR transform.structured.tile_using_for documentation"
    - "MLIR linalg matmul optimization"
    - "MLIR lowering passes reference"
    """
    # Claude Code: Use web_search tool here
    return f"Search for: {query}"

def search_optimization_techniques(topic: str) -> str:
    """
    Template function for researching optimization techniques.
    Claude Code should use web_search tool to implement this.
    
    Example searches:
    - "cache-oblivious matrix multiplication"
    - "hierarchical tiling matmul"
    - "SIMD vectorization strategies"
    """
    # Claude Code: Use web_search tool here
    return f"Search for: {topic}"

def search_github_examples(query: str) -> str:
    """
    Template function for finding GitHub examples.
    Claude Code should use web_search tool to implement this.
    
    Example searches:
    - "site:github.com/llvm/llvm-project MLIR matmul transform"
    - "MLIR linalg tiling example github"
    """
    # Claude Code: Use web_search tool here
    return f"Search GitHub for: {query}"

def debug_with_search(error_message: str) -> str:
    """
    Template function for debugging errors via search.
    Claude Code should use web_search tool to implement this.
    
    Example searches:
    - Search for the exact error message
    - "MLIR [specific error] solution"
    - "site:discourse.llvm.org [error keywords]"
    """
    # Claude Code: Use web_search tool here
    return f"Search for error: {error_message}"

def main():
    """Main optimization loop - customize this for your strategy"""
    optimizer = MatmulOptimizer()
    
    # TODO: Claude Code - Start by researching MLIR optimization
    # Use web_search to:
    # 1. Learn about MLIR transform dialect
    # 2. Find matmul optimization examples
    # 3. Understand common optimization techniques
    
    # Example searches to start with:
    # - "MLIR transform dialect matmul optimization tutorial"
    # - "MLIR linalg tiling vectorization examples"
    # - "site:github.com/llvm/llvm-project matmul transform"
    
    # Discover all matmul operations
    matmul_indices = optimizer.discover_matmuls()
    print(f"Found matmul operations: {matmul_indices}")
    
    # TODO: Claude Code should implement the optimization strategy here
    # This is a template showing the workflow with web search integration
    
    for matmul_idx in matmul_indices:
        print(f"\n{'#'*60}")
        print(f"# Optimizing Matmul {matmul_idx}")
        print(f"{'#'*60}\n")
        
        max_iterations = 50
        no_improvement_count = 0
        target_slowdown = 0.5
        
        for i in range(max_iterations):
            iteration = optimizer.get_next_iteration(matmul_idx)
            
            # TODO: Generate schedule and passes based on:
            # 1. Analysis of previous results
            # 2. Examination of lowered code
            # 3. Optimization strategies from research
            
            # Claude Code: Use web_search when needed:
            # - If stuck: search for new optimization techniques
            # - If errors: search for error messages and solutions
            # - If unsure: search for MLIR documentation
            # - For inspiration: search for academic papers on matmul
            
            # Example web search integration:
            # if iteration == 1:
            #     # Research initial strategies
            #     search_mlir_docs("transform.structured.tile_using_for")
            #     search_optimization_techniques("hierarchical tiling matrix")
            # 
            # if no_improvement_count > 2:
            #     # Look for new ideas
            #     search_github_examples("MLIR matmul high performance")
            #     search_optimization_techniques("register blocking matmul")
            
            # Example placeholder:
            schedule_content = "// TODO: Generate transform dialect schedule"
            passes_content = "// TODO: Generate lowering passes"
            strategy = "TODO: Describe optimization strategy"
            
            # Create the schedule files
            schedule_name = optimizer.create_schedule(
                matmul_idx, iteration, schedule_content, passes_content
            )
            
            # Run benchmark
            result = optimizer.submit_benchmark(matmul_idx, schedule_name)
            
            if result is None:
                print(f"Skipping iteration {iteration} due to benchmark failure")
                # TODO: Claude Code - Debug with web search
                # debug_with_search("MLIR benchmark failed")
                continue
            
            # Log the result
            optimizer.log_result(matmul_idx, iteration, result, strategy)
            
            # Check if target met
            if result.slowdown_vs_pytorch < target_slowdown:
                print(f"🎉 TARGET MET for Matmul {matmul_idx}! "
                      f"Slowdown: {result.slowdown_vs_pytorch:.4f}x")
                break
            
            # Check for improvement
            best = optimizer.get_best_result(matmul_idx)
            if best and best["iteration"] != iteration:
                if result.slowdown_vs_pytorch >= best["slowdown_vs_pytorch"]:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
            
            # If stuck, search for new ideas
            if no_improvement_count >= 3:
                print("No improvement for 3 iterations. Searching for new techniques...")
                # TODO: Claude Code - Use web_search here
                # search_optimization_techniques("advanced matmul optimization")
                # search_github_examples("MLIR high performance matmul")
            
            if no_improvement_count >= 5:
                print(f"No improvement for 5 iterations. Moving to next matmul.")
                break
            
            # Optionally analyze lowered code for insights
            if i % 5 == 0:  # Every 5 iterations
                if optimizer.run_lower(matmul_idx, schedule_name):
                    outputs = optimizer.analyze_lowered_output()
                    # TODO: Analyze outputs to guide next iteration
                    # If you see issues in the assembly, search for solutions
    
    # Generate final report
    optimizer.generate_report()
    print("\n✅ Optimization complete!")

if __name__ == "__main__":
    main()
