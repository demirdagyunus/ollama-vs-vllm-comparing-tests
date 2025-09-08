"""
Model: Qwen3-4B
Deployment: Ollama ile Docker containerları ile production-ready setup  
Test Yapısı: LangGraph ile çok adımlı reasoning agent'ı
Task Türleri:
    - Mathematical reasoning (çok adımlı matematik problemleri)
    - Code generation ve debugging
    - Multi-hop question answering
Paralel istekler: 20-50 eş zamanlı karmaşık görev
Süre: Her test seviyesi için 15 dakika (complex tasks için daha uzun)
Metrikler:
    - Task completion time
    - Accuracy (Doğru cevap oranı)
    - Memory consumption per request
    - Queue waiting time
    - Reasoning steps quality
    
Ortak İzleme Metrikleri:

* Sistem Metrikleri:
    - CPU, RAM, GPU utilization
    - Disk I/O, Network I/O
    - Container resource limits

* Uygulama Metrikleri:
    - Request latency distribution
    - Error rates ve türleri
    - Queue depth ve waiting time
    - Cache hit rates (varsa)

* Kalite Metrikleri:
    - Response accuracy (human evaluation sample)
    - Consistency across multiple runs
    - Token generation quality
    - Reasoning chain quality
"""

import asyncio
import aiohttp
import time
import json
import statistics
import psutil
import GPUtil
import threading
import random
import re
import tracemalloc
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import subprocess
import os
import requests
from queue import Queue, Empty
import ollama

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TestConfig:
    """Test configuration for complex reasoning tasks with Ollama"""
    model_name: str = "qwen3:4b"
    base_url: str = "http://localhost:11434"
    deployment_type: str = "ollama"
    test_levels: List[int] = None
    test_duration: int = 60  # 1 minute for debugging
    warmup_duration: int = 10  # 10 seconds warmup for debugging
    cooldown_duration: int = 60  # 1 minute cooldown
    max_tokens: int = 512  # Increased for complex reasoning
    temperature: float = 0.1  # Lower temperature for more consistent reasoning
    api_key: Optional[str] = None
    enable_accuracy_testing: bool = True
    enable_memory_profiling: bool = True

    def __post_init__(self):
        if self.test_levels is None:
            self.test_levels = [2, 5]  # Small test levels for debugging


@dataclass 
class ComplexTaskMetrics:
    """Metrics for complex reasoning tasks"""
    task_id: str
    task_type: str  # math, code, qa
    timestamp: float
    completion_time: float
    success: bool
    accuracy_score: float  # 0-1 scale
    reasoning_steps: int
    memory_usage_mb: float
    queue_wait_time: float
    error_message: Optional[str] = None
    response_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0
    reasoning_quality: float = 0.0  # 0-1 scale


@dataclass
class ComplexTestResult:
    """Test results for complex reasoning scenario"""
    level: int
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    avg_completion_time: float
    p95_completion_time: float
    p99_completion_time: float
    min_completion_time: float
    max_completion_time: float
    overall_accuracy: float
    math_accuracy: float
    code_accuracy: float  
    qa_accuracy: float
    avg_reasoning_steps: float
    avg_memory_per_request: float
    peak_memory_usage: float
    avg_queue_wait_time: float
    max_queue_wait_time: float
    throughput: float
    error_rate: float
    avg_cpu_usage: float
    avg_memory_usage: float
    avg_gpu_usage: float
    avg_gpu_memory: float
    peak_cpu_usage: float
    peak_memory_usage_system: float
    peak_gpu_usage: float
    peak_gpu_memory: float
    test_duration: float
    avg_tokens_per_second: float
    total_tokens_generated: int
    completion_time_distribution: Dict[str, float]
    task_type_distribution: Dict[str, int]
    error_types: Dict[str, int]
    reasoning_quality_stats: Dict[str, float]


class QueueMonitor:
    """Monitor request queue depth and waiting times"""
    
    def __init__(self):
        self.queue_depths = []
        self.wait_times = []
        self.monitoring = False
        self.current_queue_depth = 0
        self.lock = threading.Lock()
    
    def add_to_queue(self) -> float:
        """Add request to queue and return queue entry time"""
        with self.lock:
            self.current_queue_depth += 1
            self.queue_depths.append(self.current_queue_depth)
            return time.time()
    
    def remove_from_queue(self, entry_time: float):
        """Remove request from queue and record wait time"""
        with self.lock:
            self.current_queue_depth = max(0, self.current_queue_depth - 1)
            wait_time = time.time() - entry_time
            self.wait_times.append(wait_time)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get queue statistics"""
        with self.lock:
            return {
                "avg_queue_depth": statistics.mean(self.queue_depths) if self.queue_depths else 0,
                "max_queue_depth": max(self.queue_depths) if self.queue_depths else 0,
                "avg_wait_time": statistics.mean(self.wait_times) if self.wait_times else 0,
                "max_wait_time": max(self.wait_times) if self.wait_times else 0,
                "total_requests": len(self.wait_times)
            }


class ResourceMonitor:
    """Real-time system resource monitoring"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.cpu_readings = []
        self.memory_readings = []
        self.gpu_readings = []
        self.gpu_memory_readings = []
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        """Start resource monitoring in a separate thread"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.cpu_readings.clear()
        self.memory_readings.clear()
        self.gpu_readings.clear()
        self.gpu_memory_readings.clear()
        
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Resource monitoring started")

    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Resource monitoring stopped")

    def _monitor_resources(self):
        """Internal monitoring loop"""
        while self.monitoring:
            try:
                # CPU and Memory
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_percent = psutil.virtual_memory().percent
                
                self.cpu_readings.append(cpu_percent)
                self.memory_readings.append(memory_percent)

                # GPU monitoring
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Use first GPU
                        self.gpu_readings.append(gpu.load * 100)
                        self.gpu_memory_readings.append(gpu.memoryUtil * 100)
                    else:
                        self.gpu_readings.append(0)
                        self.gpu_memory_readings.append(0)
                except Exception as e:
                    logger.warning(f"GPU monitoring failed: {e}")
                    self.gpu_readings.append(0)
                    self.gpu_memory_readings.append(0)

            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")

            time.sleep(self.monitoring_interval)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive resource usage statistics"""
        def safe_stats(readings, default=0):
            if not readings:
                return {"avg": default, "max": default, "min": default}
            return {
                "avg": statistics.mean(readings),
                "max": max(readings),
                "min": min(readings)
            }

        return {
            "cpu": safe_stats(self.cpu_readings),
            "memory": safe_stats(self.memory_readings),
            "gpu": safe_stats(self.gpu_readings),
            "gpu_memory": safe_stats(self.gpu_memory_readings)
        }


class ComplexReasoningAgent:
    """LangGraph-inspired complex reasoning agent for various task types"""
    
    @staticmethod
    def get_mathematical_problems() -> List[Tuple[str, str, float]]:
        """Get mathematical reasoning problems with expected answers and difficulty scores"""
        return [
            (
                "A store has 150 items. On Monday, 1/3 of the items were sold. On Tuesday, 2/5 of the remaining items were sold. How many items are left?",
                "60",
                0.7
            ),
            (
                "If f(x) = 2x² + 3x - 1, what is f(4) + f(-2)?",
                "32",
                0.8
            ),
            (
                "A train travels 240 km in 3 hours. If it increases its speed by 20%, how long will it take to travel 400 km?",
                "4.17 hours",
                0.9
            ),
            (
                "Solve the system: 2x + 3y = 11, 4x - y = 2. What is x + y?",
                "4",
                0.8
            ),
            (
                "A rectangular garden has perimeter 60m and area 200m². What are its dimensions?",
                "20m × 10m",
                0.85
            ),
            (
                "If log₂(x) + log₂(x+6) = 4, what is x?",
                "2",
                0.9
            ),
            (
                "A company's profit increases by 15% each year. If the profit was $80,000 in 2020, what will it be in 2024?",
                "$139,980",
                0.75
            ),
            (
                "Find the derivative of f(x) = x³ - 4x² + 5x - 2 at x = 3",
                "14",
                0.8
            ),
            (
                "A ball is thrown upward with initial velocity 20 m/s. When will it reach maximum height? (g = 9.8 m/s²)",
                "2.04 seconds",
                0.85
            ),
            (
                "Calculate the compound interest on $5000 at 8% per annum for 3 years compounded quarterly.",
                "$6346.93",
                0.8
            )
        ]
    
    @staticmethod
    def get_code_problems() -> List[Tuple[str, str, float]]:
        """Get code generation and debugging problems"""
        return [
            (
                "Write a Python function to find the longest common subsequence of two strings. Include error handling.",
                "def lcs(s1, s2):",
                0.85
            ),
            (
                "Debug this Python code and explain the issue:\n```\ndef factorial(n):\n    if n = 1:\n        return 1\n    return n * factorial(n-1)\n```",
                "== instead of =",
                0.7
            ),
            (
                "Create a binary search algorithm that works on a rotated sorted array. Handle edge cases.",
                "def search_rotated",
                0.9
            ),
            (
                "Write a function to validate a binary search tree. Include comprehensive test cases.",
                "def is_valid_bst",
                0.8
            ),
            (
                "Implement a thread-safe singleton pattern in Python with lazy initialization.",
                "threading.Lock",
                0.85
            ),
            (
                "Fix this memory leak in the following code:\n```\nclass Node:\n    def __init__(self, data):\n        self.data = data\n        self.parent = None\n        self.children = []\n```",
                "circular reference",
                0.8
            ),
            (
                "Write a function to merge k sorted linked lists efficiently. Analyze time complexity.",
                "O(n log k)",
                0.9
            ),
            (
                "Create a decorator that caches function results with TTL (time-to-live) functionality.",
                "@lru_cache",
                0.75
            ),
            (
                "Implement a rate limiter using the token bucket algorithm in Python.",
                "token bucket",
                0.85
            ),
            (
                "Write a function to detect cycles in a directed graph using DFS. Handle disconnected components.",
                "DFS cycle detection",
                0.9
            )
        ]
    
    @staticmethod
    def get_multihop_qa_problems() -> List[Tuple[str, str, float]]:
        """Get multi-hop question answering problems"""
        return [
            (
                "Who was the president of the United States when the Berlin Wall fell, and what major economic policy did he implement during his presidency?",
                "George H.W. Bush",
                0.8
            ),
            (
                "What programming language was created by the founder of Python, and in which year was the first version released?",
                "Python by Guido van Rossum in 1991",
                0.7
            ),
            (
                "Which company developed the first commercial microprocessor, and what was the clock speed of their 8008 processor?",
                "Intel, 740 kHz",
                0.9
            ),
            (
                "Who directed the movie that won the Academy Award for Best Picture in 1994, and what was their previous Oscar-winning film?",
                "Robert Zemeckis, Forrest Gump",
                0.85
            ),
            (
                "What is the capital of the country where CERN is located, and in which year was the World Wide Web invented there?",
                "Bern, Switzerland, 1989",
                0.8
            ),
            (
                "Which mathematical concept was developed by the mathematician who also created the foundations of computer science theory?",
                "Alan Turing - Turing machines",
                0.9
            ),
            (
                "What disease did the scientist who discovered penicillin later work on, and which Nobel Prize did he win?",
                "Alexander Fleming, Nobel Prize in Physiology or Medicine",
                0.85
            ),
            (
                "In which city was the company founded that created the first successful electric car to achieve mass production in the 21st century?",
                "San Carlos, California (Tesla)",
                0.8
            ),
            (
                "Who wrote the novel that was adapted into the film that won the most Academy Awards in history, and how many Oscars did it win?",
                "J.R.R. Tolkien, Lord of the Rings, 11 Oscars",
                0.9
            ),
            (
                "Which element was discovered by the scientist who also discovered radioactivity, and what is its atomic number?",
                "Marie Curie, Radium, atomic number 88",
                0.85
            )
        ]
    
    @staticmethod
    def get_random_task() -> Tuple[str, str, str, float]:
        """Get a random task from all categories"""
        task_types = {
            "math": ComplexReasoningAgent.get_mathematical_problems(),
            "code": ComplexReasoningAgent.get_code_problems(), 
            "qa": ComplexReasoningAgent.get_multihop_qa_problems()
        }
        
        task_type = random.choice(list(task_types.keys()))
        problem, expected_answer, difficulty = random.choice(task_types[task_type])
        return task_type, problem, expected_answer, difficulty
    
    @staticmethod
    def evaluate_response_accuracy(task_type: str, response: str, expected_answer: str) -> float:
        """Evaluate response accuracy based on task type"""
        if not response or not expected_answer:
            return 0.0
        
        response_lower = response.lower().strip()
        expected_lower = expected_answer.lower().strip()
        
        if task_type == "math":
            # Extract numbers from response for mathematical problems
            response_numbers = re.findall(r'-?\d+\.?\d*', response)
            expected_numbers = re.findall(r'-?\d+\.?\d*', expected_answer)
            
            if response_numbers and expected_numbers:
                try:
                    resp_val = float(response_numbers[-1])  # Last number in response
                    exp_val = float(expected_numbers[-1])   # Last number in expected
                    # Allow 5% tolerance for numerical answers
                    if abs(resp_val - exp_val) / max(abs(exp_val), 1) <= 0.05:
                        return 1.0
                    elif abs(resp_val - exp_val) / max(abs(exp_val), 1) <= 0.1:
                        return 0.7
                    else:
                        return 0.3
                except:
                    pass
        
        elif task_type == "code":
            # Check for key programming concepts and syntax
            key_terms = expected_answer.split()
            found_terms = sum(1 for term in key_terms if term.lower() in response_lower)
            if found_terms == len(key_terms):
                return 1.0
            elif found_terms >= len(key_terms) * 0.7:
                return 0.8
            elif found_terms >= len(key_terms) * 0.5:
                return 0.6
            else:
                return 0.3
        
        elif task_type == "qa":
            # Check for key information in multi-hop QA
            key_terms = expected_answer.split()
            found_terms = sum(1 for term in key_terms if term.lower() in response_lower)
            if found_terms >= len(key_terms) * 0.8:
                return 1.0
            elif found_terms >= len(key_terms) * 0.6:
                return 0.7
            elif found_terms >= len(key_terms) * 0.4:
                return 0.5
            else:
                return 0.2
        
        # Fallback: simple string similarity
        if expected_lower in response_lower:
            return 1.0
        elif any(word in response_lower for word in expected_lower.split()):
            return 0.5
        else:
            return 0.0
    
    @staticmethod
    def evaluate_reasoning_quality(response: str, task_type: str) -> float:
        """Evaluate the quality of reasoning in the response"""
        if not response:
            return 0.0
        
        response_lower = response.lower()
        reasoning_indicators = {
            "math": ["because", "therefore", "since", "step", "first", "then", "next", "finally", "solve", "calculate"],
            "code": ["function", "return", "if", "else", "loop", "algorithm", "complexity", "handle", "error", "test"],
            "qa": ["according to", "based on", "research shows", "evidence", "source", "study", "found", "discovered"]
        }
        
        indicators = reasoning_indicators.get(task_type, [])
        found_indicators = sum(1 for indicator in indicators if indicator in response_lower)
        
        # Basic scoring based on reasoning indicators and response length
        indicator_score = min(found_indicators / len(indicators), 1.0) if indicators else 0.5
        length_score = min(len(response.split()) / 100, 1.0)  # Normalize by expected length
        
        return (indicator_score * 0.7) + (length_score * 0.3)


class OllamaComplexClient:
    """Ollama client optimized for complex reasoning tasks"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        # Create Ollama client
        self.client = ollama.Client(host=config.base_url)
    
    def send_complex_request(self, task_type: str, prompt: str, task_id: str) -> Tuple[bool, float, Optional[str], int, int, int, float]:
        """
        Send complex reasoning request to Ollama
        Returns: (success, completion_time, response_text, prompt_tokens, completion_tokens, total_tokens, memory_usage_mb)
        """
        # Start memory profiling for this request
        if self.config.enable_memory_profiling:
            tracemalloc.start()
        start_time = time.time()
        
        try:
            # Create detailed prompt for complex reasoning
            system_prompt = self._get_system_prompt(task_type)
            full_prompt = f"{system_prompt}\n\nTask: {prompt}\n\nPlease provide a detailed step-by-step solution:"
            
            logger.debug(f"Sending request to Ollama for task {task_id} with type {task_type}")
            
            # Ollama chat completion
            response = self.client.chat(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            )
            
            end_time = time.time()
            completion_time = end_time - start_time
            
            # Get memory usage
            memory_usage_mb = 0
            if self.config.enable_memory_profiling:
                try:
                    current, peak = tracemalloc.get_traced_memory()
                    memory_usage_mb = peak / 1024 / 1024
                    tracemalloc.stop()
                except:
                    memory_usage_mb = 0
            
            # Extract response text
            response_text = response.get("message", {}).get("content", "")
            
            # Estimate token usage (Ollama doesn't provide exact tokens)
            prompt_tokens = len(full_prompt.split()) * 1.3  # rough token estimation
            completion_tokens = len(response_text.split()) * 1.3 if response_text else 0
            total_tokens = int(prompt_tokens + completion_tokens)
            
            return True, completion_time, response_text, int(prompt_tokens), int(completion_tokens), total_tokens, memory_usage_mb
                
        except Exception as e:
            end_time = time.time()
            completion_time = end_time - start_time
            if self.config.enable_memory_profiling:
                try:
                    tracemalloc.stop()
                except:
                    pass
            logger.error(f"Complex task request failed: {e}")
            return False, completion_time, str(e), 0, 0, 0, 0
    
    def _get_system_prompt(self, task_type: str) -> str:
        """Get system prompt optimized for each task type"""
        prompts = {
            "math": "You are an expert mathematician. Solve problems step-by-step, showing all work clearly. Provide numerical answers when requested.",
            "code": "You are an expert software engineer. Write clean, efficient code with proper error handling. Explain your reasoning and approach.",
            "qa": "You are a knowledgeable research assistant. Answer questions using logical reasoning and provide detailed explanations with supporting facts."
        }
        return prompts.get(task_type, "You are a helpful assistant. Provide detailed, accurate responses.")
    
    def check_health(self) -> bool:
        """Check if Ollama service is healthy"""
        try:
            logger.info(f"Checking health of Ollama service")
            # Try to list models to check if service is available
            models = self.client.list()
            logger.info(f"Health check successful, found {len(models.get('models', []))} models")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


class ComplexPerformanceTester:
    """Performance tester for complex reasoning scenarios"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results: List[ComplexTestResult] = []
        self.queue_monitor = QueueMonitor()
    
    def check_service_health(self) -> bool:
        """Check if Ollama service is available"""
        client = OllamaComplexClient(self.config)
        return client.check_health()
    
    def run_warmup(self) -> None:
        """Run warmup with complex reasoning tasks"""
        logger.info(f"Starting complex reasoning warmup for {self.config.warmup_duration} seconds...")
        
        client = OllamaComplexClient(self.config)
        warmup_end = time.time() + self.config.warmup_duration
        warmup_count = 0
        
        while time.time() < warmup_end:
            task_type, prompt, expected_answer, difficulty = ComplexReasoningAgent.get_random_task()
            success, _, _, _, _, _, _ = client.send_complex_request(task_type, prompt, f"warmup_{warmup_count}")
            warmup_count += 1
            if success:
                logger.debug(f"Warmup task {warmup_count} ({task_type}) completed")
            time.sleep(2)  # Longer delay for complex tasks
        
        logger.info(f"Complex reasoning warmup completed with {warmup_count} tasks")
    
    def run_single_complex_task(self, client: OllamaComplexClient, task_id: str) -> ComplexTaskMetrics:
        """Run a single complex reasoning task"""
        # Queue monitoring
        queue_entry_time = self.queue_monitor.add_to_queue()
        
        task_type, prompt, expected_answer, difficulty = ComplexReasoningAgent.get_random_task()
        timestamp = time.time()
        
        success, completion_time, response, prompt_tokens, completion_tokens, total_tokens, memory_usage = \
            client.send_complex_request(task_type, prompt, task_id)
        
        # Remove from queue
        self.queue_monitor.remove_from_queue(queue_entry_time)
        queue_wait_time = queue_entry_time - timestamp if queue_entry_time > timestamp else 0
        
        # Evaluate accuracy and reasoning quality
        accuracy_score = 0.0
        reasoning_quality = 0.0
        reasoning_steps = 0
        
        if success and response:
            if self.config.enable_accuracy_testing:
                accuracy_score = ComplexReasoningAgent.evaluate_response_accuracy(task_type, response, expected_answer)
                reasoning_quality = ComplexReasoningAgent.evaluate_reasoning_quality(response, task_type)
                # Count reasoning steps (rough estimate based on sentence count)
                reasoning_steps = len([s for s in response.split('.') if len(s.strip()) > 10])
        
        return ComplexTaskMetrics(
            task_id=task_id,
            task_type=task_type,
            timestamp=timestamp,
            completion_time=completion_time,
            success=success,
            accuracy_score=accuracy_score,
            reasoning_steps=reasoning_steps,
            memory_usage_mb=memory_usage,
            queue_wait_time=queue_wait_time,
            error_message=response if not success else None,
            response_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
            reasoning_quality=reasoning_quality
        )
    
    def run_test_level(self, concurrent_users: int) -> ComplexTestResult:
        """Run complex reasoning test for a specific concurrency level"""
        logger.info(f"Starting complex reasoning test with {concurrent_users} concurrent users for {self.config.test_duration} seconds")
        
        # Start resource monitoring
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        # Initialize metrics collection
        all_metrics: List[ComplexTaskMetrics] = []
        test_start_time = time.time()
        test_end_time = test_start_time + self.config.test_duration
        
        logger.info("Complex reasoning test metrics initialized")
        
        # Worker function for complex reasoning
        def complex_worker(worker_id: int) -> List[ComplexTaskMetrics]:
            logger.info(f"Complex reasoning worker {worker_id} started")
            client = OllamaComplexClient(self.config)
            worker_metrics = []
            task_count = 0
            
            logger.info(f"Complex reasoning worker {worker_id} started")
            
            while time.time() < test_end_time:
                try:
                    task_id = f"worker_{worker_id}_task_{task_count}"
                    metrics = self.run_single_complex_task(client, task_id)
                    worker_metrics.append(metrics)
                    task_count += 1
                    
                    if task_count % 5 == 0:
                        logger.debug(f"Worker {worker_id}: {task_count} complex tasks completed")
                    
                except Exception as e:
                    logger.error(f"Complex worker {worker_id} error: {e}")
                    worker_metrics.append(ComplexTaskMetrics(
                        task_id=f"worker_{worker_id}_error_{task_count}",
                        task_type="error",
                        timestamp=time.time(),
                        completion_time=0,
                        success=False,
                        accuracy_score=0.0,
                        reasoning_steps=0,
                        memory_usage_mb=0,
                        queue_wait_time=0,
                        error_message=str(e)
                    ))
                
                # Small delay for complex reasoning tasks
                time.sleep(0.1)
            
            logger.info(f"Complex reasoning worker {worker_id} completed with {task_count} tasks")
            return worker_metrics
        
        # Run workers in parallel
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(complex_worker, i) for i in range(concurrent_users)]
            
            for future in as_completed(futures):
                try:
                    worker_metrics = future.result()
                    all_metrics.extend(worker_metrics)
                except Exception as e:
                    logger.error(f"Complex worker failed: {e}")
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        actual_test_duration = time.time() - test_start_time
        
        logger.info(f"Complex reasoning test level {concurrent_users} completed. Processing {len(all_metrics)} metrics...")
        
        return self._calculate_complex_test_result(concurrent_users, all_metrics, monitor, actual_test_duration)
    
    def _calculate_complex_test_result(self, level: int, metrics: List[ComplexTaskMetrics], 
                                     monitor: ResourceMonitor, test_duration: float) -> ComplexTestResult:
        """Calculate comprehensive test results for complex reasoning tasks"""
        
        successful_metrics = [m for m in metrics if m.success]
        failed_metrics = [m for m in metrics if not m.success]
        
        total_tasks = len(metrics)
        successful_tasks = len(successful_metrics)
        failed_tasks = len(failed_metrics)
        
        # Completion time calculations
        if successful_metrics:
            completion_times = [m.completion_time for m in successful_metrics]
            avg_completion_time = statistics.mean(completion_times)
            p95_completion_time = np.percentile(completion_times, 95)
            p99_completion_time = np.percentile(completion_times, 99)
            min_completion_time = min(completion_times)
            max_completion_time = max(completion_times)
            
            completion_time_distribution = {
                "p50": np.percentile(completion_times, 50),
                "p75": np.percentile(completion_times, 75),
                "p90": np.percentile(completion_times, 90),
                "p95": p95_completion_time,
                "p99": p99_completion_time,
                "p99.9": np.percentile(completion_times, 99.9)
            }
        else:
            avg_completion_time = p95_completion_time = p99_completion_time = 0
            min_completion_time = max_completion_time = 0
            completion_time_distribution = {}
        
        # Accuracy calculations
        if successful_metrics:
            overall_accuracy = statistics.mean([m.accuracy_score for m in successful_metrics])
            
            # Task-specific accuracy
            math_metrics = [m for m in successful_metrics if m.task_type == "math"]
            code_metrics = [m for m in successful_metrics if m.task_type == "code"]
            qa_metrics = [m for m in successful_metrics if m.task_type == "qa"]
            
            math_accuracy = statistics.mean([m.accuracy_score for m in math_metrics]) if math_metrics else 0
            code_accuracy = statistics.mean([m.accuracy_score for m in code_metrics]) if code_metrics else 0
            qa_accuracy = statistics.mean([m.accuracy_score for m in qa_metrics]) if qa_metrics else 0
            
            # Reasoning statistics
            avg_reasoning_steps = statistics.mean([m.reasoning_steps for m in successful_metrics])
            reasoning_qualities = [m.reasoning_quality for m in successful_metrics if m.reasoning_quality > 0]
            reasoning_quality_stats = {
                "avg": statistics.mean(reasoning_qualities) if reasoning_qualities else 0,
                "min": min(reasoning_qualities) if reasoning_qualities else 0,
                "max": max(reasoning_qualities) if reasoning_qualities else 0
            }
        else:
            overall_accuracy = math_accuracy = code_accuracy = qa_accuracy = 0
            avg_reasoning_steps = 0
            reasoning_quality_stats = {"avg": 0, "min": 0, "max": 0}
        
        # Memory usage calculations
        memory_usages = [m.memory_usage_mb for m in successful_metrics if m.memory_usage_mb > 0]
        avg_memory_per_request = statistics.mean(memory_usages) if memory_usages else 0
        peak_memory_usage = max(memory_usages) if memory_usages else 0
        
        # Queue statistics
        queue_stats = self.queue_monitor.get_statistics()
        avg_queue_wait_time = queue_stats["avg_wait_time"]
        max_queue_wait_time = queue_stats["max_wait_time"]
        
        # Throughput and error rate
        throughput = successful_tasks / test_duration if test_duration > 0 else 0
        error_rate = (failed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # Token statistics
        total_tokens = sum(m.total_tokens for m in successful_metrics)
        avg_tokens_per_second = total_tokens / test_duration if test_duration > 0 else 0
        
        # Task type distribution
        task_type_distribution = {}
        for m in metrics:
            task_type_distribution[m.task_type] = task_type_distribution.get(m.task_type, 0) + 1
        
        # Error types
        error_types = {}
        for m in failed_metrics:
            error_msg = m.error_message or "Unknown error"
            error_types[error_msg] = error_types.get(error_msg, 0) + 1
        
        # Resource usage statistics
        resource_stats = monitor.get_statistics()
        
        return ComplexTestResult(
            level=level,
            total_tasks=total_tasks,
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            avg_completion_time=avg_completion_time,
            p95_completion_time=p95_completion_time,
            p99_completion_time=p99_completion_time,
            min_completion_time=min_completion_time,
            max_completion_time=max_completion_time,
            overall_accuracy=overall_accuracy,
            math_accuracy=math_accuracy,
            code_accuracy=code_accuracy,
            qa_accuracy=qa_accuracy,
            avg_reasoning_steps=avg_reasoning_steps,
            avg_memory_per_request=avg_memory_per_request,
            peak_memory_usage=peak_memory_usage,
            avg_queue_wait_time=avg_queue_wait_time,
            max_queue_wait_time=max_queue_wait_time,
            throughput=throughput,
            error_rate=error_rate,
            avg_cpu_usage=resource_stats["cpu"]["avg"],
            avg_memory_usage=resource_stats["memory"]["avg"],
            avg_gpu_usage=resource_stats["gpu"]["avg"],
            avg_gpu_memory=resource_stats["gpu_memory"]["avg"],
            peak_cpu_usage=resource_stats["cpu"]["max"],
            peak_memory_usage_system=resource_stats["memory"]["max"],
            peak_gpu_usage=resource_stats["gpu"]["max"],
            peak_gpu_memory=resource_stats["gpu_memory"]["max"],
            test_duration=test_duration,
            avg_tokens_per_second=avg_tokens_per_second,
            total_tokens_generated=total_tokens,
            completion_time_distribution=completion_time_distribution,
            task_type_distribution=task_type_distribution,
            error_types=error_types,
            reasoning_quality_stats=reasoning_quality_stats
        )
    
    def run_all_tests(self) -> List[ComplexTestResult]:
        """Run all complex reasoning test levels"""
        logger.info("=" * 80)
        logger.info("STARTING OLLAMA COMPLEX REASONING PERFORMANCE TESTS")
        logger.info("=" * 80)
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Deployment: {self.config.deployment_type}")
        logger.info(f"Base URL: {self.config.base_url}")
        logger.info(f"Test levels: {self.config.test_levels}")
        logger.info(f"Test duration per level: {self.config.test_duration} seconds")
        logger.info(f"Task types: Mathematical reasoning, Code generation/debugging, Multi-hop QA")
        
        # Check service health
        if not self.check_service_health():
            logger.error("Ollama service is not available. Please check the service.")
            return []
        
        logger.info("✓ Ollama service is healthy")
        
        # Run warmup
        self.run_warmup()
        
        results = []
        
        for i, level in enumerate(self.config.test_levels):
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"COMPLEX REASONING TEST {i+1}/{len(self.config.test_levels)}: {level} CONCURRENT USERS")
                logger.info(f"{'='*60}")
                
                result = self.run_test_level(level)
                results.append(result)
                
                logger.info(f"Complex reasoning test level {level} completed:")
                logger.info(f"  - Total tasks: {result.total_tasks}")
                logger.info(f"  - Successful: {result.successful_tasks}")
                logger.info(f"  - Failed: {result.failed_tasks}")
                logger.info(f"  - Throughput: {result.throughput:.2f} tasks/s")
                logger.info(f"  - Avg completion time: {result.avg_completion_time:.1f}s")
                logger.info(f"  - Overall accuracy: {result.overall_accuracy*100:.1f}%")
                logger.info(f"  - Error rate: {result.error_rate:.2f}%")
                
                # Cooldown between tests
                if i < len(self.config.test_levels) - 1:
                    logger.info(f"Cooldown for {self.config.cooldown_duration} seconds...")
                    time.sleep(self.config.cooldown_duration)
                
            except Exception as e:
                logger.error(f"Complex reasoning test level {level} failed: {e}")
                continue
        
        logger.info("\n" + "="*80)
        logger.info("ALL COMPLEX REASONING TESTS COMPLETED")
        logger.info("="*80)
        
        return results
    
    def generate_report(self, results: List[ComplexTestResult]) -> str:
        """Generate comprehensive complex reasoning test report"""
        report = []
        
        # Header
        report.append("=" * 100)
        report.append("OLLAMA COMPLEX REASONING PERFORMANCE TEST REPORT - QWEN3-4B")
        report.append("=" * 100)
        report.append(f"Model: {self.config.model_name}")
        report.append(f"Deployment: {self.config.deployment_type}")
        report.append(f"Base URL: {self.config.base_url}")
        report.append(f"Test Duration per Level: {self.config.test_duration} seconds ({self.config.test_duration//60} minutes)")
        report.append(f"Max Tokens: {self.config.max_tokens}")
        report.append(f"Temperature: {self.config.temperature}")
        report.append(f"Task Types: Mathematical Reasoning, Code Generation/Debugging, Multi-hop QA")
        report.append(f"Test Timestamp: {datetime.now().isoformat()}")
        report.append("")
        
        if not results:
            report.append("No test results available.")
            return "\n".join(report)
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 50)
        total_tasks = sum(r.total_tasks for r in results)
        total_successful = sum(r.successful_tasks for r in results)
        overall_accuracy = statistics.mean([r.overall_accuracy for r in results if r.overall_accuracy > 0])
        max_throughput = max(r.throughput for r in results)
        
        report.append(f"Total Complex Tasks Processed: {total_tasks:,}")
        report.append(f"Overall Success Rate: {(total_successful/total_tasks*100):.2f}%" if total_tasks > 0 else "N/A")
        report.append(f"Average Accuracy Score: {overall_accuracy*100:.2f}%")
        report.append(f"Maximum Throughput: {max_throughput:.2f} tasks/s")
        report.append("")
        
        # Performance Summary Table
        report.append("COMPLEX REASONING PERFORMANCE SUMMARY")
        report.append("-" * 140)
        report.append(f"{'Level':<8} {'Total':<8} {'Success':<8} {'Failed':<7} {'Avg(s)':<8} {'P95(s)':<8} "
                      f"{'TPS':<8} {'Accuracy%':<10} {'Math%':<7} {'Code%':<7} {'QA%':<7} {'Mem(MB)':<8}")
        report.append("-" * 140)
        
        for result in results:
            report.append(f"{result.level:<8} {result.total_tasks:<8} {result.successful_tasks:<8} "
                          f"{result.failed_tasks:<7} {result.avg_completion_time:<8.1f} "
                          f"{result.p95_completion_time:<8.1f} {result.throughput:<8.1f} "
                          f"{result.overall_accuracy*100:<10.1f} {result.math_accuracy*100:<7.1f} "
                          f"{result.code_accuracy*100:<7.1f} {result.qa_accuracy*100:<7.1f} "
                          f"{result.avg_memory_per_request:<8.1f}")
        
        report.append("")
        
        # Task Type Analysis
        report.append("TASK TYPE PERFORMANCE ANALYSIS")
        report.append("-" * 80)
        for result in results:
            report.append(f"Level {result.level}:")
            for task_type, count in result.task_type_distribution.items():
                report.append(f"  - {task_type.capitalize()}: {count} tasks")
            report.append("")
        
        # Queue and Memory Analysis
        report.append("QUEUE AND MEMORY ANALYSIS")
        report.append("-" * 80)
        report.append(f"{'Level':<8} {'Avg Queue Wait(s)':<18} {'Max Queue Wait(s)':<18} {'Peak Memory(MB)':<15}")
        report.append("-" * 80)
        
        for result in results:
            report.append(f"{result.level:<8} {result.avg_queue_wait_time:<18.2f} "
                          f"{result.max_queue_wait_time:<18.2f} {result.peak_memory_usage:<15.1f}")
        
        report.append("")
        
        # Complex Reasoning Recommendations
        report.append("OLLAMA COMPLEX REASONING SPECIFIC RECOMMENDATIONS")
        report.append("-" * 50)
        
        best_accuracy_result = max(results, key=lambda r: r.overall_accuracy)
        best_throughput_result = max(results, key=lambda r: r.throughput)
        
        report.append(f"• Best Overall Accuracy: {best_accuracy_result.level} concurrent users "
                      f"({best_accuracy_result.overall_accuracy*100:.1f}%)")
        report.append(f"• Best Throughput: {best_throughput_result.level} concurrent users "
                      f"({best_throughput_result.throughput:.2f} tasks/s)")
        
        # Performance analysis
        if any(r.avg_memory_per_request > 100 for r in results):
            report.append("• High memory usage per request detected - consider optimizing model parameters")
        
        if any(r.avg_queue_wait_time > 5 for r in results):
            report.append("• High queue wait times detected - consider increasing Ollama concurrency settings")
        
        report.append("• For complex reasoning optimization, consider:")
        report.append("  - Adjusting OLLAMA_NUM_PARALLEL for better concurrency")
        report.append("  - Setting OLLAMA_MAX_LOADED_MODELS to optimize memory usage")
        report.append("  - Using temperature values between 0.1-0.3 for more consistent reasoning")
        report.append("  - Implementing request batching for similar task types")
        
        report.append("")
        report.append("Complex reasoning test completed successfully!")
        report.append("=" * 100)
        
        return "\n".join(report)
    
    def save_results(self, results: List[ComplexTestResult], output_dir: str = "results"):
        """Save complex reasoning test results"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save text report
        report = self.generate_report(results)
        report_file = os.path.join(output_dir, f"ollama_complex_reasoning_report_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Complex reasoning report saved to {report_file}")
        
        # Save JSON data
        json_file = os.path.join(output_dir, f"ollama_complex_reasoning_data_{timestamp}.json")
        json_data = {
            "config": asdict(self.config),
            "timestamp": datetime.now().isoformat(),
            "results": [asdict(result) for result in results]
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Complex reasoning JSON data saved to {json_file}")
        
        return report_file, json_file


def main():
    """Main execution function for complex reasoning tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ollama Complex Reasoning Performance Test - Qwen3-4B")
    parser.add_argument("--model", default="qwen3:4b", help="Model name")
    parser.add_argument("--url", default="http://localhost:11435", help="Ollama base URL")
    parser.add_argument("--levels", nargs="+", type=int, default=[20, 50, 100],
                        help="Concurrency levels for complex reasoning tests")
    parser.add_argument("--duration", type=int, default=60,
                        help="Test duration per level in seconds (default: 60 = 1 minute for debugging)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup duration in seconds")
    parser.add_argument("--output-dir", default="results/scenario-2",
                        help="Output directory for results")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Maximum tokens per response for complex tasks")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for complex reasoning (lower = more consistent)")
    parser.add_argument("--disable-accuracy", action="store_true",
                        help="Disable accuracy testing to speed up tests")
    parser.add_argument("--disable-memory-profiling", action="store_true",
                        help="Disable memory profiling")
    
    args = parser.parse_args()
    
    # Create test configuration
    config = TestConfig(
        model_name=args.model,
        base_url=args.url,
        test_levels=args.levels,
        test_duration=args.duration,
        warmup_duration=args.warmup,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        enable_accuracy_testing=not args.disable_accuracy,
        enable_memory_profiling=not args.disable_memory_profiling
    )
    
    # Log configuration
    logger.info("Ollama Complex Reasoning Test Configuration:")
    for key, value in asdict(config).items():
        logger.info(f"  {key}: {value}")
    
    # Create and run tester
    tester = ComplexPerformanceTester(config)
    
    try:
        results = tester.run_all_tests()
        
        if results:
            # Save results
            report_file, json_file = tester.save_results(results, args.output_dir)
            
            # Print summary
            print("\n" + "="*80)
            print("OLLAMA COMPLEX REASONING TEST SUMMARY")
            print("="*80)
            print(f"Reports saved to:")
            print(f"  - Text report: {report_file}")
            print(f"  - JSON data: {json_file}")
            print(f"Total test levels completed: {len(results)}")
            print("="*80)
        else:
            logger.error("No complex reasoning test results generated")
            
    except KeyboardInterrupt:
        logger.info("Complex reasoning test interrupted by user")
    except Exception as e:
        logger.error(f"Complex reasoning test failed: {e}")
        raise


if __name__ == "__main__":
    main()
