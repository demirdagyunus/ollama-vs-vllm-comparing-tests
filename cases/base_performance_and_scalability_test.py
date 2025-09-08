"""

Model: Qwen3-4B
Deployment: Docker container, production-ready
Test Structure: send 10,50,100,200 parallel requests with LangGraph Agents
Time: 10 minutes for each test level
Metrics:
    - Response latency (average, p95 p99)
    - Throughput (requests/second)
    - Resource usage (CPU, Memory, GPU)
    - Error rate
LangGraph Agent: Basic Q&A (without RAG)

"""
"""
LLM Performance Test Script for Qwen3-4B
Supports both Ollama and vLLM deployments with LangGraph Agents
"""

import asyncio
import aiohttp
import time
import json
import statistics
import psutil
import GPUtil
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import argparse
import ollama
from concurrent.futures import ThreadPoolExecutor
import numpy as np


@dataclass
class TestConfig:
    model_name: str = "qwen3:4b"
    base_url: str = "http://localhost:11435"
    deployment_type: str = "ollama"
    test_levels: List[int] = None
    test_duration: int = 600  # 10 minutes in seconds

    def __post_init__(self):
        if self.test_levels is None:
            self.test_levels = [10, 50, 100, 200]


@dataclass
class TestResult:
    level: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency: float
    p95_latency: float
    p99_latency: float
    throughput: float
    error_rate: float
    avg_cpu_usage: float
    avg_memory_usage: float
    avg_gpu_usage: float
    avg_gpu_memory: float


class ResourceMonitor:
    def __init__(self):
        self.cpu_readings = []
        self.memory_readings = []
        self.gpu_readings = []
        self.gpu_memory_readings = []
        self.monitoring = False

    def start_monitoring(self):
        self.monitoring = True
        self.cpu_readings.clear()
        self.memory_readings.clear()
        self.gpu_readings.clear()
        self.gpu_memory_readings.clear()

        while self.monitoring:
            # CPU and Memory Monitoring
            self.cpu_readings.append(psutil.cpu_percent(interval=None))
            self.memory_readings.append(psutil.virtual_memory().percent)

            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    self.gpu_readings.append(gpu.load * 100)
                    self.gpu_memory_readings.append(gpu.memoryUtil * 100)
                else:
                    self.gpu_readings.append(0)
                    self.gpu_memory_readings.append(0)
            except Exception:
                self.gpu_readings.append(0)
                self.gpu_memory_readings.append(0)
            
            time.sleep(1)

    def stop_monitoring(self):
        self.monitoring = False

    def get_averages(self):
        return {
            "cpu": statistics.mean(self.cpu_readings) if self.cpu_readings else 0,
            "memory": statistics.mean(self.memory_readings) if self.memory_readings else 0,
            "gpu": statistics.mean(self.gpu_readings) if self.gpu_readings else 0,
            "gpu_memory": statistics.mean(self.gpu_memory_readings) if self.gpu_memory_readings else 0
        }


class LLMClient:
    def __init__(self, config: TestConfig):
        self.config = config
        self.session = None
        self.client = ollama.Client(host=self.config.base_url)

    def send_request_ollama(self, prompt: str) -> tuple[bool, float]:
        """Send request to Ollama"""
        start_time = time.time()
        try:
            payload = {
                "model": self.config.model_name,
                "messages": {
                    "role": "user",
                    "content": prompt
                },
                "stream": False
            }

            response = self.client.chat(model=self.config.model_name,
                                        messages=[{"role": "user", "content": prompt}])
            end_time = time.time()
            print(f"Response status: {response.status}, latency: {end_time - start_time}")
            return True, end_time - start_time
        except Exception as e:
            end_time = time.time()
            print(f"Request failed: {e}")
            return False, end_time - start_time

    def send_request_vllm(self, prompt: str) -> tuple[bool, float]:
        """Send request to VLLM"""
        start_time = time.time()
        try:
            payload = {
                "model": self.config.model_name,
                "prompt": prompt,
                "max_tokens": 150,
                "temperature": 0.7
            }

            with self.session.post(f"{self.config.base_url}/v1/completions",
                                         json=payload,
                                         headers={"Content-Type": "application/json"}) as response:
                end_time = time.time()
                print(f"Response status: {response.status}, latency: {end_time - start_time}")
                return response.status == 200, end_time - start_time
        except Exception as e:
            end_time = time.time()
            print(f"Request failed: {e}")
            return False, end_time - start_time

    def send_request(self, prompt: str) -> tuple[bool, float]:
        """Send request based on deployment type"""
        if self.config.deployment_type.lower() == "ollama":
            return self.send_request_ollama(prompt)
        elif self.config.deployment_type.lower() == "vllm":
            return self.send_request_vllm(prompt)
        else:
            raise ValueError(f"Unsupported deployment type: {self.config.deployment_type}")


class LangGraphAgent:
    """Simple LangGraph Agent for Q&A without RAG"""

    @staticmethod
    def get_test_prompts() -> List[str]:
        """Generate test prompts for Q&A"""
        return [
            "What is artificial intelligence?",
            "Explain the concept of machine learning in simple terms.",
            "What are the benefits of using Docker containers?",
            "How does neural network training work?",
            "What is the difference between supervised and unsupervised learning?",
            "Explain the concept of natural language processing.",
            "What are the main components of a computer system?",
            "How do databases work?",
            "What is cloud computing and its advantages?",
            "Explain the basics of cybersecurity.",
            "What is the difference between AI and machine learning?",
            "How do recommendation systems work?",
            "What are microservices and their benefits?",
            "Explain the concept of big data analytics.",
            "What is quantum computing?",
            "How do search engines work?",
            "What is blockchain technology?",
            "Explain the concept of DevOps.",
            "What are the principles of software engineering?",
            "How does internet routing work?"
        ]


class PerformanceTester:
    def __init__(self, config: TestConfig):
        self.config = config
        self.results = []

    def run_test_level(self, concurrent_users: int) -> TestResult:
        """Run test for specific concurrency level"""
        print(f"Starting test with {concurrent_users} concurrent users")

        monitor = ResourceMonitor()
        monitor.start_monitoring()

        # Test metrics
        latencies = []
        successful_requests = 0
        failed_requests = 0
        start_time = time.time()
        end_time = start_time + self.config.test_duration

        # Get test prompts
        prompts = LangGraphAgent.get_test_prompts()

        with LLMClient(self.config) as client:
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(concurrent_users)

            def worker():
                nonlocal successful_requests, failed_requests
                request_count = 0
                while time.time() < end_time:
                    with semaphore:
                        import random
                        prompt = random.choice(prompts)

                        success, latency = client.send_request(prompt)
                        print(f"Request {request_count} successful: {success}, latency: {latency}")
                        latencies.append(latency)

                        if success:
                            successful_requests += 1
                        else:
                            failed_requests += 1

                        request_count += 1

                        # Small delay to prevent overwhelming
                        time.sleep(1)

                return request_count

            # Start worker tasks
            tasks = [worker() for _ in range(concurrent_users)]

            # Wait for test duration
            time.sleep(self.config.test_duration)

            # Cancel tasks after test duration
            for task in tasks:
                task.cancel()

        monitor.stop_monitoring()
        monitor_task.cancel()

        # Calculate metrics
        total_requests = successful_requests + failed_requests
        actual_duration = time.time() - start_time

        if latencies:
            avg_latency = statistics.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p_99_latency = np.percentile(latencies, 99)
        else:
            avg_latency = p95_latency = p_99_latency = 0

        throughput = successful_requests / actual_duration if actual_duration > 0 else 0
        error_rate = (failed_requests / total_requests) * 100 if total_requests > 0 else 0

        resource_averages = monitor.get_averages()

        result = TestResult(
            level=concurrent_users,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_latency=avg_latency,
            p95_latency=p95_latency,
            p99_latency=p_99_latency,
            throughput=throughput,
            error_rate=error_rate,
            avg_cpu_usage=resource_averages["cpu"],
            avg_memory_usage=resource_averages["memory"],
            avg_gpu_usage=resource_averages["gpu"],
            avg_gpu_memory=resource_averages["gpu_memory"]
        )

        print(
            f"Test completed: {concurrent_users} users, {total_requests} requests, {throughput:.2f} req/s, {error_rate:.2f}% error rate")

        return result

    def run_all_tests(self) -> List[TestResult]:
        """Run all test levels"""
        print(f"Starting performance tests for {self.config.deployment_type} deployment")
        print(f"Model: {self.config.model_name}")
        print(f"Base URL: {self.config.base_url}")
        print(f"Test levels: {self.config.test_levels}")
        print(f"Test duration per level: {self.config.test_duration} seconds")

        results = []

        for level in self.config.test_levels:
            try:
                result = self.run_test_level(level)
                results.append(result)

                # Brief pause between tests
                print("Waiting 30 seconds before next test level...")
                time.sleep(30)

            except Exception as e:
                print(f"Test level {level} failed: {e}")
                continue

        return results

    def generate_report(self, results: List[TestResult]) -> str:
        """Generate comprehensive test report"""
        report = []
        report.append("=" * 80)
        report.append("LLM PERFORMANCE TEST REPORT")
        report.append("=" * 80)
        report.append(f"Model: {self.config.model_name}")
        report.append(f"Deployment: {self.config.deployment_type}")
        report.append(f"Base URL: {self.config.base_url}")
        report.append(f"Test Duration per Level: {self.config.test_duration} seconds")
        report.append(f"Timestamp: {datetime.now().isoformat()}")
        report.append("")

        # Summary table
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 80)
        report.append(f"{'Level':<8} {'Total':<8} {'Success':<8} {'Failed':<8} {'Avg(ms)':<10} "
                      f"{'P95(ms)':<10} {'P99(ms)':<10} {'RPS':<8} {'Error%':<8}")
        report.append("-" * 80)

        for result in results:
            report.append(f"{result.level:<8} {result.total_requests:<8} {result.successful_requests:<8} "
                          f"{result.failed_requests:<8} {result.avg_latency * 1000:<10.1f} "
                          f"{result.p95_latency * 1000:<10.1f} {result.p99_latency * 1000:<10.1f} "
                          f"{result.throughput:<8.1f} {result.error_rate:<8.1f}")

        report.append("")

        # Resource usage
        report.append("RESOURCE USAGE")
        report.append("-" * 60)
        report.append(f"{'Level':<8} {'CPU%':<10} {'Memory%':<12} {'GPU%':<10} {'GPU Mem%':<12}")
        report.append("-" * 60)

        for result in results:
            report.append(f"{result.level:<8} {result.avg_cpu_usage:<10.1f} "
                          f"{result.avg_memory_usage:<12.1f} {result.avg_gpu_usage:<10.1f} "
                          f"{result.avg_gpu_memory:<12.1f}")

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="LLM Performance Test")
    parser.add_argument("--model", default="qwen3:4b", help="Model name")
    parser.add_argument("--url", default="http://localhost:11436", help="Base URL")
    parser.add_argument("--deployment", choices=["ollama", "vllm"], default="ollama",
                        help="Deployment type")
    parser.add_argument("--levels", nargs="+", type=int, default=[10, 50, 100, 200],
                        help="Concurrency levels to test")
    parser.add_argument("--duration", type=int, default=600,
                        help="Test duration per level in seconds")
    parser.add_argument("--output", default="performance_report.txt",
                        help="Output report file")

    args = parser.parse_args()

    config = TestConfig(
        model_name=args.model,
        base_url=args.url,
        deployment_type=args.deployment,
        test_levels=args.levels,
        test_duration=args.duration
    )

    def run_tests():
        tester = PerformanceTester(config)
        results = tester.run_all_tests()

        # Generate and save report
        report = tester.generate_report(results)
        print(report)

        with open(args.output, 'w') as f:
            f.write(report)

        print(f"Report saved to {args.output}")

        # Also save as JSON for further analysis
        json_output = args.output.replace('.txt', '.json')
        json_data = [
            {
                'level': r.level,
                'total_requests': r.total_requests,
                'successful_requests': r.successful_requests,
                'failed_requests': r.failed_requests,
                'avg_latency_ms': r.avg_latency * 1000,
                'p95_latency_ms': r.p95_latency * 1000,
                'p99_latency_ms': r.p99_latency * 1000,
                'throughput_rps': r.throughput,
                'error_rate_percent': r.error_rate,
                'avg_cpu_usage_percent': r.avg_cpu_usage,
                'avg_memory_usage_percent': r.avg_memory_usage,
                'avg_gpu_usage_percent': r.avg_gpu_usage,
                'avg_gpu_memory_percent': r.avg_gpu_memory
            }
            for r in results
        ]

        with open(json_output, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"JSON report saved to {json_output}")

    # Run tests
    run_tests()


if __name__ == "__main__":
    main()
