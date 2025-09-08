"""
Model: Qwen3-4B
Deployment: Docker containerları ile production-ready setup
Test Yapısı: Langgraph agent'ları ile 10, 50, 100, 200 paralel istek gönderme
Süre: Her test seviyesi için 10 dakika
Metrikler:
    - Response latency (ortalama, p95, p99)
    - Throughput (requests/second)
    - Resource usage (CPU, Memory, GPU)
    - Error rate
LangGraph Agent: Basit Q&A agentı (RAG olmadan)

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
"""

import asyncio
import time
import json
import statistics
import psutil
import GPUtil
import threading
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import ollama
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import subprocess
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TestConfig:
    """Test configuration parameters"""
    model_name: str = "qwen3:4b"
    base_url: str = "http://localhost:11436"
    deployment_type: str = "ollama"
    test_levels: List[int] = None
    test_duration: int = 600  # 10 minutes in seconds
    warmup_duration: int = 60  # 1 minute warmup
    cooldown_duration: int = 30  # 30 seconds cooldown
    max_tokens: int = 150
    temperature: float = 0.7

    def __post_init__(self):
        if self.test_levels is None:
            self.test_levels = [10, 50, 100, 200]


@dataclass
class RequestMetrics:
    """Individual request metrics"""
    timestamp: float
    latency: float
    success: bool
    error_message: Optional[str] = None
    response_tokens: int = 0
    prompt_tokens: int = 0


@dataclass
class TestResult:
    """Comprehensive test results for a single concurrency level"""
    level: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency: float
    p95_latency: float
    p99_latency: float
    min_latency: float
    max_latency: float
    throughput: float
    error_rate: float
    avg_cpu_usage: float
    avg_memory_usage: float
    avg_gpu_usage: float
    avg_gpu_memory: float
    peak_cpu_usage: float
    peak_memory_usage: float
    peak_gpu_usage: float
    peak_gpu_memory: float
    test_duration: float
    avg_tokens_per_second: float
    total_tokens_generated: int
    latency_distribution: Dict[str, float]
    error_types: Dict[str, int]


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


class LangGraphAgent:
    """Simple LangGraph Agent for Q&A without RAG"""
    
    @staticmethod
    def get_test_prompts() -> List[str]:
        """Generate diverse test prompts for Q&A testing"""
        return [
            "What is artificial intelligence and how does it work?",
            "Explain the concept of machine learning in simple terms.",
            "What are the benefits of using Docker containers in production?",
            "How does neural network training work step by step?",
            "What is the difference between supervised and unsupervised learning?",
            "Explain natural language processing and its applications.",
            "What are the main components of a modern computer system?",
            "How do relational databases work and what are their advantages?",
            "What is cloud computing and what are its main benefits?",
            "Explain the fundamental principles of cybersecurity.",
            "What is the difference between AI, machine learning, and deep learning?",
            "How do recommendation systems work in practice?",
            "What are microservices and what benefits do they provide?",
            "Explain the concept of big data analytics and its importance.",
            "What is quantum computing and how is it different from classical computing?",
            "How do modern search engines work to find relevant results?",
            "What is blockchain technology and what are its use cases?",
            "Explain the DevOps methodology and its core principles.",
            "What are the key principles of good software engineering?",
            "How does internet routing work at a high level?",
            "What is containerization and why is it useful?",
            "Explain the concept of API design and best practices.",
            "What are design patterns in software development?",
            "How does caching improve application performance?",
            "What is the difference between SQL and NoSQL databases?",
            "Explain the concept of distributed systems and their challenges.",
            "What is load balancing and why is it important?",
            "How does version control work in software development?",
            "What are the principles of clean code and maintainable software?",
            "Explain the concept of continuous integration and deployment."
        ]

    @staticmethod
    def get_random_prompt() -> str:
        """Get a random test prompt"""
        return random.choice(LangGraphAgent.get_test_prompts())


class OllamaClient:
    """Ollama client for LLM requests"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.client = ollama.Client(host=config.base_url)

    def send_request(self, prompt: str) -> Tuple[bool, float, Optional[str], int]:
        """
        Send request to Ollama
        Returns: (success, latency, response_text, token_count)
        """
        start_time = time.time()
        try:
            response = self.client.chat(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_predict": self.config.max_tokens,
                    "temperature": self.config.temperature
                }
            )
            
            end_time = time.time()
            latency = end_time - start_time
            
            response_text = response['message']['content']
            # Estimate token count (rough approximation)
            token_count = len(response_text.split())
            
            return True, latency, response_text, token_count
            
        except Exception as e:
            end_time = time.time()
            latency = end_time - start_time
            logger.error(f"Request failed: {e}")
            return False, latency, str(e), 0


class PerformanceTester:
    """Main performance testing class"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results: List[TestResult] = []

    def run_warmup(self) -> None:
        """Run warmup requests to prepare the model"""
        logger.info(f"Starting warmup for {self.config.warmup_duration} seconds...")
        
        client = OllamaClient(self.config)
        warmup_end = time.time() + self.config.warmup_duration
        warmup_count = 0
        
        while time.time() < warmup_end:
            prompt = LangGraphAgent.get_random_prompt()
            success, _, _, _ = client.send_request(prompt)
            warmup_count += 1
            if success:
                logger.debug(f"Warmup request {warmup_count} completed")
            time.sleep(1)  # Small delay between warmup requests
        
        logger.info(f"Warmup completed with {warmup_count} requests")

    def run_single_request(self, client: OllamaClient, request_id: int) -> RequestMetrics:
        """Run a single request and return metrics"""
        prompt = LangGraphAgent.get_random_prompt()
        timestamp = time.time()
        
        success, latency, response, token_count = client.send_request(prompt)
        
        return RequestMetrics(
            timestamp=timestamp,
            latency=latency,
            success=success,
            error_message=response if not success else None,
            response_tokens=token_count if success else 0,
            prompt_tokens=len(prompt.split()) if success else 0
        )

    def run_test_level(self, concurrent_users: int) -> TestResult:
        """Run performance test for a specific concurrency level"""
        logger.info(f"Starting test with {concurrent_users} concurrent users for {self.config.test_duration} seconds")
        
        # Start resource monitoring
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        # Initialize metrics collection
        all_metrics: List[RequestMetrics] = []
        test_start_time = time.time()
        test_end_time = test_start_time + self.config.test_duration
        
        # Worker function for each thread
        def worker(worker_id: int) -> List[RequestMetrics]:
            client = OllamaClient(self.config)
            worker_metrics = []
            request_count = 0
            
            logger.info(f"Worker {worker_id} started")
            
            while time.time() < test_end_time:
                try:
                    metrics = self.run_single_request(client, request_count)
                    worker_metrics.append(metrics)
                    request_count += 1
                    
                    if request_count % 10 == 0:
                        logger.debug(f"Worker {worker_id}: {request_count} requests completed")
                        
                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}")
                    # Add failed request metrics
                    worker_metrics.append(RequestMetrics(
                        timestamp=time.time(),
                        latency=0,
                        success=False,
                        error_message=str(e)
                    ))
                
                # Small delay to prevent overwhelming the server
                time.sleep(0.1)
            
            logger.info(f"Worker {worker_id} completed with {request_count} requests")
            return worker_metrics

        # Run workers in parallel
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(worker, i) for i in range(concurrent_users)]
            
            # Collect results
            for future in as_completed(futures):
                try:
                    worker_metrics = future.result()
                    all_metrics.extend(worker_metrics)
                except Exception as e:
                    logger.error(f"Worker failed: {e}")

        # Stop monitoring
        monitor.stop_monitoring()
        
        # Calculate test duration
        actual_test_duration = time.time() - test_start_time
        
        logger.info(f"Test level {concurrent_users} completed. Processing {len(all_metrics)} metrics...")
        
        # Process metrics
        return self._calculate_test_result(concurrent_users, all_metrics, monitor, actual_test_duration)

    def _calculate_test_result(self, level: int, metrics: List[RequestMetrics], 
                             monitor: ResourceMonitor, test_duration: float) -> TestResult:
        """Calculate comprehensive test results from metrics"""
        
        successful_metrics = [m for m in metrics if m.success]
        failed_metrics = [m for m in metrics if not m.success]
        
        total_requests = len(metrics)
        successful_requests = len(successful_metrics)
        failed_requests = len(failed_metrics)
        
        # Latency calculations
        if successful_metrics:
            latencies = [m.latency for m in successful_metrics]
            avg_latency = statistics.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            # Latency distribution
            latency_distribution = {
                "p50": np.percentile(latencies, 50),
                "p75": np.percentile(latencies, 75),
                "p90": np.percentile(latencies, 90),
                "p95": p95_latency,
                "p99": p99_latency,
                "p99.9": np.percentile(latencies, 99.9)
            }
        else:
            avg_latency = p95_latency = p99_latency = min_latency = max_latency = 0
            latency_distribution = {}

        # Throughput calculation
        throughput = successful_requests / test_duration if test_duration > 0 else 0
        
        # Error rate
        error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0
        
        # Token statistics
        total_tokens = sum(m.response_tokens for m in successful_metrics)
        avg_tokens_per_second = total_tokens / test_duration if test_duration > 0 else 0
        
        # Error types
        error_types = {}
        for m in failed_metrics:
            error_msg = m.error_message or "Unknown error"
            error_types[error_msg] = error_types.get(error_msg, 0) + 1
        
        # Resource usage statistics
        resource_stats = monitor.get_statistics()
        
        return TestResult(
            level=level,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_latency=avg_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            min_latency=min_latency,
            max_latency=max_latency,
            throughput=throughput,
            error_rate=error_rate,
            avg_cpu_usage=resource_stats["cpu"]["avg"],
            avg_memory_usage=resource_stats["memory"]["avg"],
            avg_gpu_usage=resource_stats["gpu"]["avg"],
            avg_gpu_memory=resource_stats["gpu_memory"]["avg"],
            peak_cpu_usage=resource_stats["cpu"]["max"],
            peak_memory_usage=resource_stats["memory"]["max"],
            peak_gpu_usage=resource_stats["gpu"]["max"],
            peak_gpu_memory=resource_stats["gpu_memory"]["max"],
            test_duration=test_duration,
            avg_tokens_per_second=avg_tokens_per_second,
            total_tokens_generated=total_tokens,
            latency_distribution=latency_distribution,
            error_types=error_types
        )

    def run_all_tests(self) -> List[TestResult]:
        """Run all test levels with warmup and cooldown"""
        logger.info("=" * 80)
        logger.info("STARTING LLM PERFORMANCE TESTS")
        logger.info("=" * 80)
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Deployment: {self.config.deployment_type}")
        logger.info(f"Base URL: {self.config.base_url}")
        logger.info(f"Test levels: {self.config.test_levels}")
        logger.info(f"Test duration per level: {self.config.test_duration} seconds")
        logger.info(f"Warmup duration: {self.config.warmup_duration} seconds")
        
        # Run warmup
        self.run_warmup()
        
        results = []
        
        for i, level in enumerate(self.config.test_levels):
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"TEST LEVEL {i+1}/{len(self.config.test_levels)}: {level} CONCURRENT USERS")
                logger.info(f"{'='*60}")
                
                result = self.run_test_level(level)
                results.append(result)
                
                logger.info(f"Test level {level} completed:")
                logger.info(f"  - Total requests: {result.total_requests}")
                logger.info(f"  - Successful: {result.successful_requests}")
                logger.info(f"  - Failed: {result.failed_requests}")
                logger.info(f"  - Throughput: {result.throughput:.2f} req/s")
                logger.info(f"  - Avg latency: {result.avg_latency*1000:.1f}ms")
                logger.info(f"  - Error rate: {result.error_rate:.2f}%")
                
                # Cooldown between tests (except for the last test)
                if i < len(self.config.test_levels) - 1:
                    logger.info(f"Cooldown for {self.config.cooldown_duration} seconds...")
                    time.sleep(self.config.cooldown_duration)
                
            except Exception as e:
                logger.error(f"Test level {level} failed: {e}")
                continue
        
        logger.info("\n" + "="*80)
        logger.info("ALL TESTS COMPLETED")
        logger.info("="*80)
        
        return results

    def generate_report(self, results: List[TestResult]) -> str:
        """Generate comprehensive performance test report"""
        report = []
        
        # Header
        report.append("=" * 100)
        report.append("LLM PERFORMANCE TEST REPORT - QWEN3-4B")
        report.append("=" * 100)
        report.append(f"Model: {self.config.model_name}")
        report.append(f"Deployment: {self.config.deployment_type}")
        report.append(f"Base URL: {self.config.base_url}")
        report.append(f"Test Duration per Level: {self.config.test_duration} seconds ({self.config.test_duration//60} minutes)")
        report.append(f"Max Tokens: {self.config.max_tokens}")
        report.append(f"Temperature: {self.config.temperature}")
        report.append(f"Test Timestamp: {datetime.now().isoformat()}")
        report.append("")
        
        if not results:
            report.append("No test results available.")
            return "\n".join(report)
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 50)
        total_requests = sum(r.total_requests for r in results)
        total_successful = sum(r.successful_requests for r in results)
        overall_error_rate = ((total_requests - total_successful) / total_requests * 100) if total_requests > 0 else 0
        max_throughput = max(r.throughput for r in results)
        
        report.append(f"Total Requests Processed: {total_requests:,}")
        report.append(f"Overall Success Rate: {(total_successful/total_requests*100):.2f}%" if total_requests > 0 else "N/A")
        report.append(f"Overall Error Rate: {overall_error_rate:.2f}%")
        report.append(f"Maximum Throughput Achieved: {max_throughput:.2f} req/s")
        report.append("")
        
        # Performance Summary Table
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 120)
        report.append(f"{'Level':<8} {'Total':<8} {'Success':<8} {'Failed':<7} {'Avg(ms)':<9} {'P95(ms)':<9} "
                      f"{'P99(ms)':<9} {'RPS':<8} {'Error%':<8} {'Tokens/s':<9}")
        report.append("-" * 120)
        
        for result in results:
            report.append(f"{result.level:<8} {result.total_requests:<8} {result.successful_requests:<8} "
                          f"{result.failed_requests:<7} {result.avg_latency*1000:<9.1f} "
                          f"{result.p95_latency*1000:<9.1f} {result.p99_latency*1000:<9.1f} "
                          f"{result.throughput:<8.1f} {result.error_rate:<8.1f} {result.avg_tokens_per_second:<9.1f}")
        
        report.append("")
        
        # Resource Usage Summary
        report.append("RESOURCE USAGE SUMMARY")
        report.append("-" * 80)
        report.append(f"{'Level':<8} {'CPU%':<10} {'Memory%':<12} {'GPU%':<10} {'GPU Mem%':<12}")
        report.append("-" * 80)
        
        for result in results:
            report.append(f"{result.level:<8} {result.avg_cpu_usage:<10.1f} "
                          f"{result.avg_memory_usage:<12.1f} {result.avg_gpu_usage:<10.1f} "
                          f"{result.avg_gpu_memory:<12.1f}")
        
        report.append("")
        report.append("Test completed successfully!")
        report.append("=" * 100)
        
        return "\n".join(report)

    def save_results(self, results: List[TestResult], output_dir: str = "results"):
        """Save test results in multiple formats"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save text report
        report = self.generate_report(results)
        report_file = os.path.join(output_dir, f"ollama_performance_report_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Report saved to {report_file}")
        
        # Save JSON data for further analysis
        json_file = os.path.join(output_dir, f"ollama_performance_data_{timestamp}.json")
        json_data = {
            "config": asdict(self.config),
            "timestamp": datetime.now().isoformat(),
            "results": [asdict(result) for result in results]
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON data saved to {json_file}")
        
        return report_file, json_file


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Performance Test - Qwen3-4B")
    parser.add_argument("--model", default="qwen3:4b", help="Model name")
    parser.add_argument("--url", default="http://localhost:11436", help="Ollama base URL")
    parser.add_argument("--levels", nargs="+", type=int, default=[10, 50, 100],
                        help="Concurrency levels to test")
    parser.add_argument("--duration", type=int, default=600,
                        help="Test duration per level in seconds (default: 600 = 10 minutes)")
    parser.add_argument("--warmup", type=int, default=60,
                        help="Warmup duration in seconds")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory for results")
    parser.add_argument("--max-tokens", type=int, default=150,
                        help="Maximum tokens per response")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for text generation")
    
    args = parser.parse_args()
    
    # Create test configuration
    config = TestConfig(
        model_name=args.model,
        base_url=args.url,
        test_levels=args.levels,
        test_duration=args.duration,
        warmup_duration=args.warmup,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    # Log configuration
    logger.info("Test Configuration:")
    for key, value in asdict(config).items():
        logger.info(f"  {key}: {value}")
    
    # Create and run tester
    tester = PerformanceTester(config)
    
    try:
        results = tester.run_all_tests()
        
        if results:
            # Save results
            report_file, json_file = tester.save_results(results, args.output_dir)
            
            # Print summary
            print("\n" + "="*80)
            print("TEST SUMMARY")
            print("="*80)
            print(f"Reports saved to:")
            print(f"  - Text report: {report_file}")
            print(f"  - JSON data: {json_file}")
            print(f"Total test levels completed: {len(results)}")
            print("="*80)
        else:
            logger.error("No test results generated")
            
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()


