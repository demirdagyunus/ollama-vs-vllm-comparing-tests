"""
Model: Qwen3-4B
Deployment: vLLM ile Docker containerları ile production-ready setup
Test Yapısı: Stress testing ve load scalability with Simple Q&A LangGraph agent
Test Türleri:
    - Gradual load increase (10 → 500 concurrent requests)
    - Spike testing (sudden load bursts)
    - Endurance testing (2+ hours sustained load)
    - Breaking point identification
    - Recovery time measurement after overload
    - Graceful degradation behavior analysis
    - Memory leak detection over time

Test Profili:
    - Load Ramp-up: 10, 25, 50, 100, 200, 300, 400, 500 concurrent users
    - Spike Tests: Sudden jumps from baseline to peak load
    - Endurance: 2+ hours at optimal load level
    - Recovery: Post-overload performance measurement

Metrikler:
    - Breaking point identification (max sustainable load)
    - Recovery time after overload conditions
    - Graceful degradation behavior patterns
    - Memory leak detection and resource growth
    - Performance consistency over extended periods
    - Error rate escalation patterns
    - Queue saturation points

Ortak İzleme Metrikleri:

* Sistem Metrikleri:
    - CPU, RAM, GPU utilization over time
    - Disk I/O, Network I/O trends
    - Container resource limits and usage
    - Memory growth patterns and leak detection

* Uygulama Metrikleri:
    - Request latency distribution across load levels
    - Error rates ve türleri at each stress level
    - Queue depth ve waiting time escalation
    - Throughput degradation patterns
    - Cache hit rates (if applicable)

* Kalite Metrikleri:
    - Response accuracy consistency under stress
    - Quality degradation under high load
    - Token generation consistency
    - Reasoning chain stability
    - Error recovery capabilities
"""
import time
import statistics
import threading
import random
import json
import os
import psutil
import tracemalloc
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import requests
from collections import deque

# Try to import GPU monitoring, fallback gracefully
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    GPUtil = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class StressTestConfig:
    """Configuration for stress testing and load scalability scenarios (vLLM)"""
    model_name: str = "Qwen/Qwen3-4B"
    base_url: str = "http://localhost:8001"
    deployment_type: str = "vllm"

    # Load progression configuration
    gradual_load_levels: List[int] = None  # 10 → 500 progression
    ramp_up_duration: int = 300  # 5 minutes per level
    spike_test_duration: int = 120  # 2 minutes per spike
    endurance_duration: int = 7200  # 2 hours
    recovery_measurement_duration: int = 300  # 5 minutes

    # Spike testing configuration
    spike_baseline_load: int = 50
    spike_peak_loads: List[int] = None  # Sudden burst levels
    spike_interval: int = 600  # 10 minutes between spikes

    # Test parameters
    max_tokens: int = 256
    temperature: float = 0.3  # Lower for consistency
    warmup_duration: int = 60
    cooldown_duration: int = 120

    # Quality and monitoring
    enable_memory_leak_detection: bool = True
    enable_quality_tracking: bool = True
    enable_detailed_monitoring: bool = True
    memory_snapshot_interval: int = 60  # Every minute
    quality_sample_rate: float = 0.1  # 10% of requests for quality evaluation

    # Breaking point detection
    error_rate_threshold: float = 5.0  # 5% error rate threshold
    latency_threshold: float = 10.0  # 10s latency threshold
    memory_growth_threshold: float = 0.1  # 10% memory growth per hour threshold

    def __post_init__(self):
        if self.gradual_load_levels is None:
            self.gradual_load_levels = [10, 25, 50, 100, 200, 300, 400, 500]
        if self.spike_peak_loads is None:
            self.spike_peak_loads = [200, 400, 600, 800]


@dataclass
class StressTestMetrics:
    """Comprehensive metrics for stress testing scenarios"""
    task_id: str
    test_phase: str  # gradual, spike, endurance, recovery
    load_level: int
    timestamp: float
    completion_time: float
    tokens_per_second: float
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    success: bool
    error_message: Optional[str] = None
    memory_usage_mb: float = 0.0
    queue_wait_time: float = 0.0
    response_quality_score: float = 0.0  # 0-1 scale
    bandwidth_usage_bytes: int = 0
    thread_id: str = ""

    # Stress-specific metrics
    cpu_usage_snapshot: float = 0.0
    memory_usage_snapshot: float = 0.0
    gpu_usage_snapshot: float = 0.0
    concurrent_requests_count: int = 0
    queue_depth_snapshot: int = 0


@dataclass
class StressTestResult:
    """Results for stress testing scenario"""
    test_phase: str
    load_level: int
    duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int

    # Performance metrics
    avg_completion_time: float
    p95_completion_time: float
    p99_completion_time: float
    min_completion_time: float
    max_completion_time: float

    avg_tokens_per_second: float
    min_tokens_per_second: float
    max_tokens_per_second: float

    # Stress-specific metrics
    throughput_requests_per_second: float
    error_rate: float
    avg_queue_wait_time: float
    max_queue_wait_time: float

    # Breaking point indicators
    performance_degradation_percent: float
    memory_growth_rate_mb_per_hour: float
    peak_error_rate: float
    recovery_time_seconds: float

    # Resource usage
    avg_cpu_usage: float
    avg_memory_usage: float
    avg_gpu_usage: float
    peak_cpu_usage: float
    peak_memory_usage: float
    peak_gpu_usage: float

    # Quality metrics
    avg_response_quality: float
    quality_degradation_percent: float
    consistency_score: float

    # Detailed distributions
    completion_time_distribution: Dict[str, float]
    error_types: Dict[str, int]
    resource_usage_timeline: List[Dict[str, float]]

    # Breaking point analysis
    is_breaking_point: bool = False
    breaking_point_reason: str = ""
    recommended_max_load: int = 0


class StressTestAgent:
    """Simple Q&A LangGraph agent for stress testing"""

    @staticmethod
    def get_qa_prompts() -> List[Tuple[str, str, str]]:
        """Get Q&A prompts with expected answer types and quality criteria"""
        return [
            ("What is the capital of France?", "factual", "Paris"),
            ("Explain photosynthesis in simple terms.", "explanatory", "process plants"),
            ("What are the benefits of exercise?", "informational", "health benefits"),
            ("How do you make a basic cake?", "procedural", "ingredients steps"),
            ("What is artificial intelligence?", "conceptual", "computer systems"),
            ("Describe the water cycle.", "descriptive", "evaporation precipitation"),
            ("What causes earthquakes?", "causal", "tectonic plates"),
            ("How does the internet work?", "technical", "network protocols"),
            ("What are renewable energy sources?", "categorical", "solar wind"),
            ("Explain quantum computing basics.", "complex", "quantum bits"),
        ]

    @staticmethod
    def evaluate_response_quality(prompt: str, response: str, expected_type: str, expected_content: str) -> float:
        """Evaluate response quality for consistency tracking"""
        if not response or len(response.strip()) < 10:
            return 0.0

        response_lower = response.lower()
        expected_lower = expected_content.lower()

        # Content match
        expected_terms = expected_lower.split()
        found_terms = sum(1 for term in expected_terms if term in response_lower)
        content_score = found_terms / len(expected_terms) if expected_terms else 0.0

        type_indicators = {
            "factual": ["is", "are", "the", "located"],
            "explanatory": ["process", "when", "because", "occurs"],
            "informational": ["benefits", "advantages", "include", "such as"],
            "procedural": ["first", "then", "next", "step", "mix"],
            "conceptual": ["refers to", "concept", "idea", "definition"],
            "descriptive": ["consists of", "involves", "characterized by"],
            "causal": ["causes", "because", "due to", "results in"],
            "technical": ["system", "protocol", "network", "technology"],
            "categorical": ["types", "include", "examples", "such as"],
            "complex": ["involves", "quantum", "complex", "advanced"]
        }
        indicators = type_indicators.get(expected_type, [])
        type_score = 0.0
        if indicators:
            found_indicators = sum(1 for indicator in indicators if indicator in response_lower)
            type_score = min(found_indicators / len(indicators), 1.0)

        # Length
        length_score = 1.0
        word_count = len(response.split())
        if word_count < 20:
            length_score = word_count / 20
        elif word_count > 200:
            length_score = max(0.5, 1.0 - (word_count - 200) / 200)

        return (content_score * 0.5) + (type_score * 0.3) + (length_score * 0.2)


class MemoryLeakDetector:
    """Detect memory leaks and resource growth patterns"""

    def __init__(self, snapshot_interval: int = 60):
        self.snapshot_interval = snapshot_interval
        self.memory_snapshots = []
        self.cpu_snapshots = []
        self.gpu_snapshots = []
        self.timestamps = []
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        if self.monitoring:
            return
        self.monitoring = True
        self.memory_snapshots.clear()
        self.cpu_snapshots.clear()
        self.gpu_snapshots.clear()
        self.timestamps.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Memory leak detection monitoring started")

    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Memory leak detection monitoring stopped")

    def _monitor_resources(self):
        while self.monitoring:
            try:
                timestamp = time.time()
                memory = psutil.virtual_memory()
                memory_mb = memory.used / 1024 / 1024
                cpu_percent = psutil.cpu_percent(interval=1)
                gpu_percent = 0
                if GPU_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu_percent = gpus[0].load * 100
                    except Exception:
                        gpu_percent = 0
                self.timestamps.append(timestamp)
                self.memory_snapshots.append(memory_mb)
                self.cpu_snapshots.append(cpu_percent)
                self.gpu_snapshots.append(gpu_percent)
                max_snapshots = 4 * 60 * 60 // self.snapshot_interval
                if len(self.memory_snapshots) > max_snapshots:
                    self.memory_snapshots = self.memory_snapshots[-max_snapshots:]
                    self.cpu_snapshots = self.cpu_snapshots[-max_snapshots:]
                    self.gpu_snapshots = self.gpu_snapshots[-max_snapshots:]
                    self.timestamps = self.timestamps[-max_snapshots:]
            except Exception as e:
                logger.error(f"Error in memory leak monitoring: {e}")
            time.sleep(self.snapshot_interval)

    def analyze_memory_trends(self) -> Dict[str, Any]:
        if len(self.memory_snapshots) < 10:
            return {"status": "insufficient_data"}
        time_hours = (self.timestamps[-1] - self.timestamps[0]) / 3600
        memory_start = statistics.mean(self.memory_snapshots[:5])
        memory_end = statistics.mean(self.memory_snapshots[-5:])
        memory_growth_rate = (memory_end - memory_start) / time_hours if time_hours > 0 else 0
        x = np.array(range(len(self.memory_snapshots)))
        y = np.array(self.memory_snapshots)
        if len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            trend = "increasing" if slope > 1 else "stable" if abs(slope) <= 1 else "decreasing"
        else:
            slope = 0
            trend = "stable"
        return {
            "status": "analyzed",
            "memory_growth_rate_mb_per_hour": memory_growth_rate,
            "trend": trend,
            "slope_mb_per_snapshot": slope,
            "total_growth_mb": memory_end - memory_start,
            "duration_hours": time_hours,
            "snapshots_count": len(self.memory_snapshots),
            "current_memory_mb": self.memory_snapshots[-1] if self.memory_snapshots else 0
        }


class StressTestResourceMonitor:
    """Enhanced resource monitoring for stress testing"""

    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.cpu_readings = deque(maxlen=3600)
        self.memory_readings = deque(maxlen=3600)
        self.gpu_readings = deque(maxlen=3600)
        self.gpu_memory_readings = deque(maxlen=3600)
        self.timestamps = deque(maxlen=3600)
        self.monitoring = False
        self.monitor_thread = None
        self.lock = threading.Lock()

    def start_monitoring(self):
        if self.monitoring:
            return
        self.monitoring = True
        with self.lock:
            self.cpu_readings.clear()
            self.memory_readings.clear()
            self.gpu_readings.clear()
            self.gpu_memory_readings.clear()
            self.timestamps.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Stress test resource monitoring started")

    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=3.0)
        logger.info("Stress test resource monitoring stopped")

    def _monitor_resources(self):
        while self.monitoring:
            try:
                timestamp = time.time()
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                gpu_percent = 0
                gpu_memory_percent = 0
                if GPU_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]
                            gpu_percent = gpu.load * 100
                            gpu_memory_percent = gpu.memoryUtil * 100
                    except Exception:
                        pass
                with self.lock:
                    self.timestamps.append(timestamp)
                    self.cpu_readings.append(cpu_percent)
                    self.memory_readings.append(memory_percent)
                    self.gpu_readings.append(gpu_percent)
                    self.gpu_memory_readings.append(gpu_memory_percent)
            except Exception as e:
                logger.error(f"Error in stress test resource monitoring: {e}")
            time.sleep(self.monitoring_interval)

    def get_statistics(self) -> Dict[str, Any]:
        with self.lock:
            def safe_stats(readings, default=0):
                if not readings:
                    return {"avg": default, "max": default, "min": default, "current": default}
                readings_list = list(readings)
                return {
                    "avg": statistics.mean(readings_list),
                    "max": max(readings_list),
                    "min": min(readings_list),
                    "current": readings_list[-1] if readings_list else default
                }
            return {
                "cpu": safe_stats(self.cpu_readings),
                "memory": safe_stats(self.memory_readings),
                "gpu": safe_stats(self.gpu_readings),
                "gpu_memory": safe_stats(self.gpu_memory_readings),
                "timeline_length": len(self.timestamps),
                "monitoring_duration": (list(self.timestamps)[-1] - list(self.timestamps)[0]) if len(self.timestamps) > 1 else 0
            }

    def get_timeline_data(self) -> List[Dict[str, float]]:
        with self.lock:
            timeline = []
            timestamps_list = list(self.timestamps)
            cpu_list = list(self.cpu_readings)
            memory_list = list(self.memory_readings)
            gpu_list = list(self.gpu_readings)
            for i in range(min(len(timestamps_list), len(cpu_list), len(memory_list))):
                timeline.append({
                    "timestamp": timestamps_list[i],
                    "cpu": cpu_list[i],
                    "memory": memory_list[i],
                    "gpu": gpu_list[i] if i < len(gpu_list) else 0
                })
            return timeline


class VLLMStressTestClient:
    """vLLM client optimized for stress testing scenarios"""

    def __init__(self, config: StressTestConfig):
        self.config = config
        # Normalize localhost to 127.0.0.1 for better threading stability
        if 'localhost' in self.config.base_url:
            self.config.base_url = self.config.base_url.replace('localhost', '127.0.0.1')
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=50,
            pool_maxsize=200,
            max_retries=3,
            pool_block=False
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        self.headers = {
            "Content-Type": "application/json",
            "Connection": "keep-alive"
        }
        self.request_count = 0
        self.lock = threading.Lock()

    def send_stress_request(self, prompt: str, task_id: str) -> Tuple[bool, float, Optional[str], int, int, int, float]:
        """
        Send request optimized for stress testing (vLLM /chat)
        Returns: (success, completion_time, response_text, prompt_tokens, completion_tokens, total_tokens, memory_usage_mb)
        """
        with self.lock:
            self.request_count += 1
            current_count = self.request_count

        if self.config.enable_memory_leak_detection:
            tracemalloc.start()

        start_time = time.time()
        try:
            system_prompt = "You are a helpful assistant focused on concise, accurate answers."
            payload = {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature
            }

            response = self.session.post(
                f"{self.config.base_url}/chat",
                json=payload,
                headers=self.headers,
                timeout=60
            )

            end_time = time.time()
            completion_time = end_time - start_time

            memory_usage_mb = 0
            if self.config.enable_memory_leak_detection:
                try:
                    current, peak = tracemalloc.get_traced_memory()
                    memory_usage_mb = peak / 1024 / 1024
                    tracemalloc.stop()
                except Exception:
                    memory_usage_mb = 0

            if response.status_code == 200:
                data = response.json()
                # vLLM custom fields as used in scenario-2 script
                response_text = data.get("text", "")
                tokens_used = data.get("tokens_used", 0)
                prompt_tokens = int(len((system_prompt + "\n" + prompt).split()) * 1.3)
                completion_tokens = int(tokens_used)
                total_tokens = int(prompt_tokens + completion_tokens)
                return True, completion_time, response_text, prompt_tokens, completion_tokens, total_tokens, memory_usage_mb
            else:
                logger.error(f"Request failed [{response.status_code}]: {response.text}")
                return False, completion_time, f"HTTP {response.status_code}: {response.text}", 0, 0, 0, memory_usage_mb

        except requests.exceptions.Timeout:
            end_time = time.time()
            completion_time = end_time - start_time
            if self.config.enable_memory_leak_detection:
                try:
                    tracemalloc.stop()
                except Exception:
                    pass
            logger.error("Stress request timed out")
            return False, completion_time, "Request timeout", 0, 0, 0, 0
        except requests.exceptions.ConnectionError as e:
            end_time = time.time()
            completion_time = end_time - start_time
            if self.config.enable_memory_leak_detection:
                try:
                    tracemalloc.stop()
                except Exception:
                    pass
            # Retry hint (already normalized localhost)
            logger.error(f"Connection error: {e}")
            return False, completion_time, f"Connection error: {e}", 0, 0, 0, 0
        except Exception as e:
            end_time = time.time()
            completion_time = end_time - start_time
            if self.config.enable_memory_leak_detection:
                try:
                    tracemalloc.stop()
                except Exception:
                    pass
            logger.error(f"Stress request {current_count} failed: {e}")
            return False, completion_time, str(e), 0, 0, 0, 0

    def check_health(self) -> bool:
        """Check if vLLM service is healthy"""
        try:
            logger.info(f"Checking health at {self.config.base_url}/health")
            resp = self.session.get(f"{self.config.base_url}/health", timeout=10)
            logger.info(f"Stress test health check response: {resp.status_code}")
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Stress test health check failed: {e}")
            return False


class StressTestPerformanceTester:
    """Comprehensive performance tester for stress testing scenarios (vLLM)"""

    def __init__(self, config: StressTestConfig):
        self.config = config
        self.results: List[StressTestResult] = []
        self.memory_leak_detector = MemoryLeakDetector(config.memory_snapshot_interval)
        self.all_metrics: List[StressTestMetrics] = []

        # Breaking point tracking
        self.breaking_point_detected = False
        self.breaking_point_load = 0
        self.baseline_performance: Optional[Dict[str, float]] = None

    def check_service_health(self) -> bool:
        client = VLLMStressTestClient(self.config)
        return client.check_health()

    def run_warmup(self) -> None:
        logger.info(f"Starting stress test warmup for {self.config.warmup_duration} seconds...")
        client = VLLMStressTestClient(self.config)
        warmup_end = time.time() + self.config.warmup_duration
        warmup_count = 0
        qa_prompts = StressTestAgent.get_qa_prompts()
        while time.time() < warmup_end:
            prompt, _, _ = random.choice(qa_prompts)
            try:
                success, _, _, _, _, _, _ = client.send_stress_request(prompt, f"warmup_{warmup_count}")
                warmup_count += 1
                if success:
                    logger.debug(f"Warmup task {warmup_count} completed")
            except Exception as e:
                logger.warning(f"Warmup task failed: {e}")
            time.sleep(0.5)
        logger.info(f"Stress test warmup completed with {warmup_count} tasks")

    def run_single_stress_task(self, client: VLLMStressTestClient, task_id: str,
                               test_phase: str, load_level: int, concurrent_count: int) -> StressTestMetrics:
        qa_prompts = StressTestAgent.get_qa_prompts()
        prompt, answer_type, expected_content = random.choice(qa_prompts)
        cpu_snapshot = psutil.cpu_percent(interval=None)
        memory_snapshot = psutil.virtual_memory().percent
        gpu_snapshot = 0
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_snapshot = gpus[0].load * 100
            except Exception:
                pass
        timestamp = time.time()
        success, completion_time, response, prompt_tokens, completion_tokens, total_tokens, memory_usage = \
            client.send_stress_request(prompt, task_id)
        tokens_per_second = completion_tokens / completion_time if completion_time > 0 and completion_tokens > 0 else 0
        quality_score = 0.0
        if success and response and self.config.enable_quality_tracking and random.random() < self.config.quality_sample_rate:
            quality_score = StressTestAgent.evaluate_response_quality(prompt, response, answer_type, expected_content)
        bandwidth_usage = len(prompt.encode('utf-8')) + len(response.encode('utf-8')) if response else len(prompt.encode('utf-8'))
        return StressTestMetrics(
            task_id=task_id,
            test_phase=test_phase,
            load_level=load_level,
            timestamp=timestamp,
            completion_time=completion_time,
            tokens_per_second=tokens_per_second,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            success=success,
            error_message=response if not success else None,
            memory_usage_mb=memory_usage,
            queue_wait_time=0.0,
            response_quality_score=quality_score,
            bandwidth_usage_bytes=bandwidth_usage,
            thread_id=threading.current_thread().name,
            cpu_usage_snapshot=cpu_snapshot,
            memory_usage_snapshot=memory_snapshot,
            gpu_usage_snapshot=gpu_snapshot,
            concurrent_requests_count=concurrent_count,
            queue_depth_snapshot=0
        )

    def detect_breaking_point(self, metrics: List[StressTestMetrics], load_level: int) -> Tuple[bool, str]:
        if not metrics:
            return False, ""
        successful_metrics = [m for m in metrics if m.success]
        total_requests = len(metrics)
        successful_requests = len(successful_metrics)
        error_rate = ((total_requests - successful_requests) / total_requests * 100) if total_requests > 0 else 0
        avg_latency = statistics.mean([m.completion_time for m in successful_metrics]) if successful_metrics else 0
        breaking_reasons = []
        if error_rate > self.config.error_rate_threshold:
            breaking_reasons.append(f"Error rate {error_rate:.1f}% exceeds threshold {self.config.error_rate_threshold}%")
        if avg_latency > self.config.latency_threshold:
            breaking_reasons.append(f"Average latency {avg_latency:.1f}s exceeds threshold {self.config.latency_threshold}s")
        memory_analysis = self.memory_leak_detector.analyze_memory_trends()
        if memory_analysis.get("status") == "analyzed":
            growth_rate = memory_analysis.get("memory_growth_rate_mb_per_hour", 0)
            if growth_rate > self.config.memory_growth_threshold * 1024:
                breaking_reasons.append(f"Memory growth rate {growth_rate:.1f} MB/h indicates potential leak")
        if breaking_reasons:
            return True, "; ".join(breaking_reasons)
        return False, ""

    def run_gradual_load_test(self) -> List[StressTestResult]:
        logger.info("=" * 80)
        logger.info("STARTING GRADUAL LOAD INCREASE TEST (vLLM)")
        logger.info("=" * 80)
        results = []
        for load_level in self.config.gradual_load_levels:
            if self.breaking_point_detected:
                logger.info(f"Breaking point detected, skipping load level {load_level}")
                continue
            logger.info(f"Testing gradual load level: {load_level} concurrent users")
            monitor = StressTestResourceMonitor()
            monitor.start_monitoring()
            level_metrics: List[StressTestMetrics] = []
            test_start_time = time.time()
            test_end_time = test_start_time + self.config.ramp_up_duration

            def gradual_worker(worker_id: int) -> List[StressTestMetrics]:
                client = VLLMStressTestClient(self.config)
                worker_metrics = []
                task_count = 0
                while time.time() < test_end_time:
                    try:
                        task_id = f"gradual_{load_level}_{worker_id}_{task_count}"
                        metrics = self.run_single_stress_task(client, task_id, "gradual", load_level, load_level)
                        worker_metrics.append(metrics)
                        task_count += 1
                        time.sleep(0.1)
                    except Exception as e:
                        logger.error(f"Gradual worker {worker_id} error: {e}")
                return worker_metrics

            with ThreadPoolExecutor(max_workers=load_level) as executor:
                futures = [executor.submit(gradual_worker, i) for i in range(load_level)]
                for future in as_completed(futures):
                    try:
                        worker_metrics = future.result()
                        level_metrics.extend(worker_metrics)
                    except Exception as e:
                        logger.error(f"Gradual worker failed: {e}")

            monitor.stop_monitoring()
            actual_duration = time.time() - test_start_time
            is_breaking, breaking_reason = self.detect_breaking_point(level_metrics, load_level)
            if is_breaking:
                logger.warning(f"Breaking point detected at load level {load_level}: {breaking_reason}")
                self.breaking_point_detected = True
                self.breaking_point_load = load_level

            result = self._calculate_stress_test_result("gradual", load_level, level_metrics, monitor, actual_duration)
            result.is_breaking_point = is_breaking
            result.breaking_point_reason = breaking_reason
            result.recommended_max_load = load_level - self.config.gradual_load_levels[0] if is_breaking and load_level > self.config.gradual_load_levels[0] else load_level
            results.append(result)
            self.all_metrics.extend(level_metrics)
            if not self.baseline_performance and result.successful_requests > 0:
                self.baseline_performance = {
                    "avg_completion_time": result.avg_completion_time,
                    "throughput": result.throughput_requests_per_second,
                    "error_rate": result.error_rate,
                    "avg_quality": result.avg_response_quality
                }
            logger.info(f"Gradual load level {load_level} completed: {result.successful_requests}/{result.total_requests} successful")
            if load_level < max(self.config.gradual_load_levels):
                logger.info(f"Cooldown for {self.config.cooldown_duration} seconds...")
                time.sleep(self.config.cooldown_duration)
        return results

    def run_spike_test(self) -> List[StressTestResult]:
        logger.info("=" * 80)
        logger.info("STARTING SPIKE LOAD TEST (vLLM)")
        logger.info("=" * 80)
        results = []
        for spike_load in self.config.spike_peak_loads:
            if self.breaking_point_detected and spike_load > self.breaking_point_load:
                logger.info(f"Spike load {spike_load} exceeds breaking point, skipping")
                continue
            logger.info(f"Testing spike load: {self.config.spike_baseline_load} → {spike_load} concurrent users")
            monitor = StressTestResourceMonitor()
            monitor.start_monitoring()

            logger.info(f"Baseline phase: {self.config.spike_baseline_load} users for 60 seconds")
            baseline_metrics = self._run_load_phase("spike_baseline", self.config.spike_baseline_load, 60)

            logger.info(f"Spike phase: {spike_load} users for {self.config.spike_test_duration} seconds")
            spike_metrics = self._run_load_phase("spike_peak", spike_load, self.config.spike_test_duration)

            logger.info(f"Recovery phase: {self.config.spike_baseline_load} users for 60 seconds")
            recovery_start_time = time.time()
            recovery_metrics = self._run_load_phase("spike_recovery", self.config.spike_baseline_load, 60)
            recovery_time = time.time() - recovery_start_time

            monitor.stop_monitoring()
            all_spike_metrics = baseline_metrics + spike_metrics + recovery_metrics
            total_duration = self.config.spike_test_duration + 120

            result = self._calculate_stress_test_result("spike", spike_load, all_spike_metrics, monitor, total_duration)
            result.recovery_time_seconds = recovery_time
            results.append(result)
            self.all_metrics.extend(all_spike_metrics)
            logger.info(f"Spike test {spike_load} completed: {result.successful_requests}/{result.total_requests} successful")

            if spike_load < max(self.config.spike_peak_loads):
                logger.info(f"Waiting {self.config.spike_interval} seconds between spikes...")
                time.sleep(self.config.spike_interval)
        return results

    def run_endurance_test(self) -> StressTestResult:
        logger.info("=" * 80)
        logger.info("STARTING ENDURANCE TEST (vLLM)")
        logger.info("=" * 80)
        if self.breaking_point_detected:
            endurance_load = max(10, self.breaking_point_load // 2)
        else:
            endurance_load = self.config.gradual_load_levels[len(self.config.gradual_load_levels) // 2]
        logger.info(f"Running endurance test with {endurance_load} concurrent users for {self.config.endurance_duration//3600:.1f} hours")
        self.memory_leak_detector.start_monitoring()
        monitor = StressTestResourceMonitor()
        monitor.start_monitoring()
        endurance_metrics = self._run_load_phase("endurance", endurance_load, self.config.endurance_duration)
        monitor.stop_monitoring()
        self.memory_leak_detector.stop_monitoring()
        result = self._calculate_stress_test_result("endurance", endurance_load, endurance_metrics, monitor, self.config.endurance_duration)
        memory_analysis = self.memory_leak_detector.analyze_memory_trends()
        if memory_analysis.get("status") == "analyzed":
            result.memory_growth_rate_mb_per_hour = memory_analysis.get("memory_growth_rate_mb_per_hour", 0)
        self.all_metrics.extend(endurance_metrics)
        logger.info(f"Endurance test completed: {result.successful_requests}/{result.total_requests} successful over {result.duration/3600:.1f} hours")
        return result

    def _run_load_phase(self, phase_name: str, load_level: int, duration: int) -> List[StressTestMetrics]:
        phase_metrics: List[StressTestMetrics] = []
        test_start_time = time.time()
        test_end_time = test_start_time + duration

        def phase_worker(worker_id: int) -> List[StressTestMetrics]:
            client = VLLMStressTestClient(self.config)
            worker_metrics = []
            task_count = 0
            while time.time() < test_end_time:
                try:
                    task_id = f"{phase_name}_{load_level}_{worker_id}_{task_count}"
                    metrics = self.run_single_stress_task(client, task_id, phase_name, load_level, load_level)
                    worker_metrics.append(metrics)
                    task_count += 1
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Phase worker {worker_id} error: {e}")
            return worker_metrics

        with ThreadPoolExecutor(max_workers=load_level) as executor:
            futures = [executor.submit(phase_worker, i) for i in range(load_level)]
            for future in as_completed(futures):
                try:
                    worker_metrics = future.result()
                    phase_metrics.extend(worker_metrics)
                except Exception as e:
                    logger.error(f"Phase worker failed: {e}")

        return phase_metrics

    def _calculate_stress_test_result(self, test_phase: str, load_level: int, metrics: List[StressTestMetrics],
                                      monitor: StressTestResourceMonitor, duration: float) -> StressTestResult:
        successful_metrics = [m for m in metrics if m.success]
        failed_metrics = [m for m in metrics if not m.success]
        total_requests = len(metrics)
        successful_requests = len(successful_metrics)
        failed_requests = len(failed_metrics)

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
                "p99": p99_completion_time
            }
            tps_values = [m.tokens_per_second for m in successful_metrics if m.tokens_per_second > 0]
            avg_tps = statistics.mean(tps_values) if tps_values else 0
            min_tps = min(tps_values) if tps_values else 0
            max_tps = max(tps_values) if tps_values else 0
        else:
            avg_completion_time = p95_completion_time = p99_completion_time = 0
            min_completion_time = max_completion_time = 0
            completion_time_distribution = {}
            avg_tps = min_tps = max_tps = 0

        throughput = successful_requests / duration if duration > 0 else 0
        error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0
        avg_queue_wait = 0
        max_queue_wait = 0

        performance_degradation = 0.0
        if self.baseline_performance:
            baseline_latency = self.baseline_performance["avg_completion_time"]
            if baseline_latency > 0:
                performance_degradation = ((avg_completion_time - baseline_latency) / baseline_latency * 100)

        quality_scores = [m.response_quality_score for m in successful_metrics if m.response_quality_score > 0]
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0

        quality_degradation = 0.0
        if self.baseline_performance and self.baseline_performance.get("avg_quality", 0) > 0:
            baseline_quality = self.baseline_performance["avg_quality"]
            quality_degradation = ((baseline_quality - avg_quality) / baseline_quality * 100)

        consistency_score = 1.0
        if successful_metrics and len(successful_metrics) > 1:
            std_dev = statistics.stdev([m.completion_time for m in successful_metrics])
            mean_time = avg_completion_time
            if mean_time > 0:
                cv = std_dev / mean_time
                consistency_score = max(0, 1 - cv)

        resource_stats = monitor.get_statistics()
        timeline_data = monitor.get_timeline_data()

        error_types = {}
        for m in failed_metrics:
            error_msg = m.error_message or "Unknown error"
            if "timeout" in error_msg.lower():
                error_type = "timeout"
            elif "connection" in error_msg.lower():
                error_type = "connection"
            elif "memory" in error_msg.lower():
                error_type = "memory"
            else:
                error_type = "other"
            error_types[error_type] = error_types.get(error_type, 0) + 1

        return StressTestResult(
            test_phase=test_phase,
            load_level=load_level,
            duration=duration,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_completion_time=avg_completion_time,
            p95_completion_time=p95_completion_time,
            p99_completion_time=p99_completion_time,
            min_completion_time=min_completion_time,
            max_completion_time=max_completion_time,
            avg_tokens_per_second=avg_tps,
            min_tokens_per_second=min_tps,
            max_tokens_per_second=max_tps,
            throughput_requests_per_second=throughput,
            error_rate=error_rate,
            avg_queue_wait_time=avg_queue_wait,
            max_queue_wait_time=max_queue_wait,
            performance_degradation_percent=performance_degradation,
            memory_growth_rate_mb_per_hour=0.0,
            peak_error_rate=error_rate,
            recovery_time_seconds=0.0,
            avg_cpu_usage=resource_stats["cpu"]["avg"],
            avg_memory_usage=resource_stats["memory"]["avg"],
            avg_gpu_usage=resource_stats["gpu"]["avg"],
            peak_cpu_usage=resource_stats["cpu"]["max"],
            peak_memory_usage=resource_stats["memory"]["max"],
            peak_gpu_usage=resource_stats["gpu"]["max"],
            avg_response_quality=avg_quality,
            quality_degradation_percent=quality_degradation,
            consistency_score=consistency_score,
            completion_time_distribution=completion_time_distribution,
            error_types=error_types,
            resource_usage_timeline=timeline_data
        )

    def run_all_tests(self) -> Dict[str, Any]:
        logger.info("=" * 80)
        logger.info("STARTING VLLM STRESS TESTING & LOAD SCALABILITY TESTS")
        logger.info("=" * 80)
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Deployment: {self.config.deployment_type}")
        logger.info(f"Base URL: {self.config.base_url}")
        logger.info(f"Gradual load levels: {self.config.gradual_load_levels}")
        logger.info(f"Spike loads: {self.config.spike_peak_loads}")
        logger.info(f"Endurance duration: {self.config.endurance_duration//3600} hours")

        if not self.check_service_health():
            logger.error("vLLM service is not available. Please check the service.")
            return {"error": "Service unavailable"}

        logger.info("✓ vLLM stress test service is healthy")
        self.run_warmup()
        self.memory_leak_detector.start_monitoring()

        results = {
            "gradual_load_results": [],
            "spike_test_results": [],
            "endurance_test_result": None
        }
        try:
            logger.info("\n" + "="*60)
            logger.info("PHASE 1: GRADUAL LOAD INCREASE TEST")
            logger.info("="*60)
            gradual_results = self.run_gradual_load_test()
            results["gradual_load_results"] = gradual_results

            logger.info("\n" + "="*60)
            logger.info("PHASE 2: SPIKE LOAD TEST")
            logger.info("="*60)
            spike_results = self.run_spike_test()
            results["spike_test_results"] = spike_results

            logger.info("\n" + "="*60)
            logger.info("PHASE 3: ENDURANCE TEST")
            logger.info("="*60)
            endurance_result = self.run_endurance_test()
            results["endurance_test_result"] = endurance_result

        except Exception as e:
            logger.error(f"Stress test phase failed: {e}")
        finally:
            self.memory_leak_detector.stop_monitoring()

        logger.info("\n" + "="*80)
        logger.info("ALL STRESS TESTS COMPLETED")
        logger.info("="*80)
        return results

    def generate_report(self, results: Dict[str, Any]) -> str:
        report = []
        report.append("=" * 120)
        report.append("VLLM STRESS TESTING & LOAD SCALABILITY REPORT - QWEN3-4B")
        report.append("=" * 120)
        report.append(f"Model: {self.config.model_name}")
        report.append(f"Deployment: {self.config.deployment_type}")
        report.append(f"Base URL: {self.config.base_url}")
        report.append(f"Max Tokens: {self.config.max_tokens}")
        report.append(f"Temperature: {self.config.temperature}")
        report.append(f"Test Timestamp: {datetime.now().isoformat()}")
        report.append("")

        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 50)

        gradual_results = results.get("gradual_load_results", [])
        spike_results = results.get("spike_test_results", [])
        endurance_result = results.get("endurance_test_result")

        total_requests = sum(r.total_requests for r in gradual_results + spike_results)
        if endurance_result:
            total_requests += endurance_result.total_requests

        breaking_point_load = 0
        breaking_point_reason = "Not detected"
        for result in gradual_results:
            if result.is_breaking_point:
                breaking_point_load = result.load_level
                breaking_point_reason = result.breaking_point_reason
                break

        report.append(f"Total Requests Processed: {total_requests:,}")
        report.append(f"Breaking Point: {breaking_point_load} concurrent users" if breaking_point_load > 0 else "Breaking Point: Not detected within test range")
        report.append(f"Breaking Point Reason: {breaking_point_reason}")

        if gradual_results:
            non_breaking = [r for r in gradual_results if not r.is_breaking_point]
            if non_breaking:
                max_stable_load = max(r.load_level for r in non_breaking)
                report.append(f"Maximum Stable Load: {max_stable_load} concurrent users")
            else:
                report.append("Maximum Stable Load: Not found (all levels exceeded thresholds)")
            max_throughput = max(r.throughput_requests_per_second for r in gradual_results)
            report.append(f"Peak Throughput: {max_throughput:.2f} requests/s")

        report.append("")
        if gradual_results:
            report.append("GRADUAL LOAD TEST RESULTS")
            report.append("-" * 100)
            report.append(f"{'Load':<6} {'Requests':<9} {'Success':<9} {'Failed':<7} {'RPS':<7} {'Latency':<9} {'P95':<7} {'Error%':<7} {'Quality':<8}")
            report.append("-" * 100)
            for result in gradual_results:
                status = " *BP*" if result.is_breaking_point else ""
                report.append(f"{result.load_level:<6} {result.total_requests:<9} {result.successful_requests:<9} "
                              f"{result.failed_requests:<7} {result.throughput_requests_per_second:<7.1f} "
                              f"{result.avg_completion_time:<9.3f} {result.p95_completion_time:<7.3f} "
                              f"{result.error_rate:<7.1f} {result.avg_response_quality*100:<8.1f}{status}")
            report.append("*BP* = Breaking Point detected")
            report.append("")

        if spike_results:
            report.append("SPIKE TEST RESULTS")
            report.append("-" * 80)
            report.append(f"{'Spike Load':<12} {'Requests':<9} {'Success Rate':<13} {'Recovery Time':<15} {'Peak Error%':<12}")
            report.append("-" * 80)
            for result in spike_results:
                success_rate = (result.successful_requests / result.total_requests * 100) if result.total_requests > 0 else 0
                report.append(f"{result.load_level:<12} {result.total_requests:<9} {success_rate:<13.1f} "
                              f"{result.recovery_time_seconds:<15.1f} {result.peak_error_rate:<12.1f}")
            report.append("")

        if endurance_result:
            report.append("ENDURANCE TEST RESULTS")
            report.append("-" * 80)
            duration_hours = endurance_result.duration / 3600
            report.append(f"Duration: {duration_hours:.1f} hours")
            report.append(f"Load Level: {endurance_result.load_level} concurrent users")
            report.append(f"Total Requests: {endurance_result.total_requests:,}")
            report.append(f"Success Rate: {(endurance_result.successful_requests/endurance_result.total_requests*100):.2f}%")
            report.append(f"Average Latency: {endurance_result.avg_completion_time:.3f}s")
            report.append(f"Throughput: {endurance_result.throughput_requests_per_second:.2f} req/s")
            report.append(f"Memory Growth Rate: {endurance_result.memory_growth_rate_mb_per_hour:.1f} MB/hour")
            report.append(f"Consistency Score: {endurance_result.consistency_score*100:.1f}%")
            report.append("")

        report.append("PERFORMANCE DEGRADATION ANALYSIS")
        report.append("-" * 80)
        if gradual_results:
            for result in gradual_results:
                degradation = result.performance_degradation_percent
                quality_loss = result.quality_degradation_percent
                report.append(f"Load {result.load_level}: {degradation:+.1f}% latency change, {quality_loss:+.1f}% quality change")
        report.append("")

        report.append("RESOURCE USAGE PATTERNS")
        report.append("-" * 80)
        all_results = gradual_results + spike_results
        if endurance_result:
            all_results.append(endurance_result)
        if all_results:
            report.append(f"{'Phase':<15} {'Load':<6} {'CPU%':<8} {'Memory%':<10} {'GPU%':<8} {'Peak CPU%':<10}")
            report.append("-" * 80)
            for result in all_results:
                report.append(f"{result.test_phase:<15} {result.load_level:<6} {result.avg_cpu_usage:<8.1f} "
                              f"{result.avg_memory_usage:<10.1f} {result.avg_gpu_usage:<8.1f} {result.peak_cpu_usage:<10.1f}")
        report.append("")

        memory_analysis = self.memory_leak_detector.analyze_memory_trends()
        if memory_analysis.get("status") == "analyzed":
            report.append("MEMORY LEAK ANALYSIS")
            report.append("-" * 50)
            report.append(f"Memory Growth Rate: {memory_analysis['memory_growth_rate_mb_per_hour']:.1f} MB/hour")
            report.append(f"Total Memory Growth: {memory_analysis['total_growth_mb']:.1f} MB")
            report.append(f"Trend: {memory_analysis['trend']}")
            report.append(f"Test Duration: {memory_analysis['duration_hours']:.1f} hours")
            if memory_analysis['memory_growth_rate_mb_per_hour'] > 100:
                report.append("⚠️  WARNING: Significant memory growth detected - potential memory leak")
            elif memory_analysis['memory_growth_rate_mb_per_hour'] < 10:
                report.append("✅ Memory usage appears stable")
            else:
                report.append("ℹ️  Moderate memory growth observed - monitor in production")
            report.append("")

        report.append("STRESS TESTING RECOMMENDATIONS")
        report.append("-" * 80)
        if breaking_point_load > 0:
            safe_load = max(10, breaking_point_load - 50)
            report.append(f"• Recommended maximum load: {safe_load} concurrent users (safety margin below breaking point)")
            report.append(f"• Breaking point detected at: {breaking_point_load} users due to: {breaking_point_reason}")
        else:
            report.append("• No breaking point detected within test range - consider testing higher loads")
        if gradual_results:
            best_performance = max(gradual_results, key=lambda r: r.throughput_requests_per_second if not r.is_breaking_point else 0)
            report.append(f"• Optimal performance at: {best_performance.load_level} users ({best_performance.throughput_requests_per_second:.1f} req/s)")
        if spike_results:
            avg_recovery = statistics.mean([r.recovery_time_seconds for r in spike_results if r.recovery_time_seconds > 0]) if spike_results else 0
            report.append(f"• Average spike recovery time: {avg_recovery:.1f} seconds")
        if endurance_result:
            if endurance_result.consistency_score > 0.8:
                report.append("• System shows good consistency during extended load")
            else:
                report.append("• System shows performance variance during extended load - investigate")
        report.append("• For production deployment consider:")
        report.append("  - Implementing circuit breakers at 80% of breaking point")
        report.append("  - Setting up auto-scaling triggers before breaking point")
        report.append("  - Monitoring memory growth patterns continuously")
        report.append("  - Implementing graceful degradation strategies")
        report.append("  - Load balancing across multiple vLLM instances")
        report.append("")
        report.append("Stress testing and load scalability analysis completed!")
        report.append("=" * 120)
        return "\n".join(report)

    def save_results(self, results: Dict[str, Any], output_dir: str = "results/scenario-4"):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = self.generate_report(results)
        report_file = os.path.join(output_dir, f"vllm_stress_test_report_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Stress test report saved to {report_file}")

        json_data = {
            "config": asdict(self.config),
            "timestamp": datetime.now().isoformat(),
            "results": {
                "gradual_load_results": [asdict(r) for r in results.get("gradual_load_results", [])],
                "spike_test_results": [asdict(r) for r in results.get("spike_test_results", [])],
                "endurance_test_result": asdict(results["endurance_test_result"]) if results.get("endurance_test_result") else None
            },
            "memory_leak_analysis": self.memory_leak_detector.analyze_memory_trends(),
            "breaking_point_detected": self.breaking_point_detected,
            "breaking_point_load": self.breaking_point_load,
            "baseline_performance": self.baseline_performance,
            "total_metrics_collected": len(self.all_metrics)
        }
        json_file = os.path.join(output_dir, f"vllm_stress_test_data_{timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Stress test JSON data saved to {json_file}")
        return report_file, json_file


def main():
    """Main execution function for vLLM stress testing"""
    import argparse

    parser = argparse.ArgumentParser(description="vLLM Stress Testing & Load Scalability - Qwen3-4B")
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="Model name")
    parser.add_argument("--url", default="http://localhost:8000", help="vLLM base URL")
    parser.add_argument("--gradual-loads", nargs="+", type=int, default=[50, 200],
                        help="Gradual load increase levels")
    parser.add_argument("--spike-loads", nargs="+", type=int, default=[200, 400],
                        help="Spike test load levels")
    parser.add_argument("--ramp-duration", type=int, default=180,
                        help="Duration for each gradual load level (seconds)")
    parser.add_argument("--spike-duration", type=int, default=180,
                        help="Duration for each spike test (seconds)")
    parser.add_argument("--endurance-duration", type=int, default=1800,
                        help="Endurance test duration (seconds, default: 2 hours)")
    parser.add_argument("--output-dir", default="results/scenario-4",
                        help="Output directory for results")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Maximum tokens per response")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Temperature for consistency")
    parser.add_argument("--error-threshold", type=float, default=5.0,
                        help="Error rate threshold for breaking point detection (%)")
    parser.add_argument("--latency-threshold", type=float, default=10.0,
                        help="Latency threshold for breaking point detection (seconds)")
    parser.add_argument("--disable-memory-detection", action="store_true",
                        help="Disable memory leak detection")
    parser.add_argument("--disable-quality-tracking", action="store_true",
                        help="Disable response quality tracking")

    args = parser.parse_args()

    config = StressTestConfig(
        model_name=args.model,
        base_url=args.url,
        gradual_load_levels=args.gradual_loads,
        spike_peak_loads=args.spike_loads,
        ramp_up_duration=args.ramp_duration,
        spike_test_duration=args.spike_duration,
        endurance_duration=args.endurance_duration,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        error_rate_threshold=args.error_threshold,
        latency_threshold=args.latency_threshold,
        enable_memory_leak_detection=not args.disable_memory_detection,
        enable_quality_tracking=not args.disable_quality_tracking
    )

    logger.info("vLLM Stress Test Configuration:")
    for key, value in asdict(config).items():
        logger.info(f"  {key}: {value}")

    tester = StressTestPerformanceTester(config)
    try:
        results = tester.run_all_tests()
        if results and not results.get("error"):
            report_file, json_file = tester.save_results(results, args.output_dir)
            print("\n" + "="*100)
            print("VLLM STRESS TEST SUMMARY")
            print("="*100)
            print("Reports saved to:")
            print(f"  - Text report: {report_file}")
            print(f"  - JSON data: {json_file}")

            gradual_results = results.get("gradual_load_results", [])
            if gradual_results:
                breaking_point = next((r for r in gradual_results if r.is_breaking_point), None)
                if breaking_point:
                    print(f"Breaking point detected: {breaking_point.load_level} users ({breaking_point.breaking_point_reason})")
                else:
                    print("No breaking point detected within test range")
                max_throughput = max(r.throughput_requests_per_second for r in gradual_results)
                print(f"Peak throughput: {max_throughput:.2f} req/s")
                total_requests = sum(r.total_requests for r in gradual_results)
                total_successful = sum(r.successful_requests for r in gradual_results)
                if total_requests > 0:
                    print(f"Total requests: {total_requests:,} ({total_successful/total_requests*100:.1f}% success)")

            endurance_result = results.get("endurance_test_result")
            if endurance_result:
                print(f"Endurance test: {endurance_result.duration/3600:.1f}h, {endurance_result.memory_growth_rate_mb_per_hour:.1f} MB/h growth")
            print("="*100)
        else:
            logger.error("No stress test results generated or error occurred")
    except KeyboardInterrupt:
        logger.info("Stress test interrupted by user")
    except Exception as e:
        logger.error(f"Stress test failed: {e}")
        raise


if __name__ == "__main__":
    main()