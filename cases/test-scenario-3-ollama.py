"""
Model: Qwen3-4B
Deployment: Ollama ile Docker containerları ile production-ready setup  
Test Yapısı: LangGraph ile streaming conversational agent'ı
Test Türleri:
    - Streaming response (token-by-token)
    - Batch responses (complete response) 
    - Real-time conversation simulation
    - Interactive dialogue sessions
Paralel istekler: 50-150 eş zamanlı conversation
Süre: Her test seviyesi için 15 dakika (streaming için daha uzun)
Metrikler:
    - Time to first token (TTFT)
    - Tokens per second
    - User experience metrics (perceived latency)
    - Bandwidth usage
    - Conversation flow quality
    - Response coherence and context awareness
    
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
    - Conversation context retention
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
import ollama

# Try to import GPU monitoring, fallback gracefully
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    GPUtil = None

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class StreamingTestConfig:
    """Test configuration for streaming and conversational scenarios with Ollama"""
    model_name: str = "qwen3:4b"
    base_url: str = "http://localhost:11434"
    deployment_type: str = "ollama"
    test_levels: List[int] = None
    test_duration: int = 900  # 15 minutes as specified in docstring
    warmup_duration: int = 30  # 30 seconds warmup for streaming tests
    cooldown_duration: int = 60  # 1 minute cooldown
    max_tokens: int = 512  # Increased for better conversation quality
    temperature: float = 0.7  # Higher temperature for more natural conversation
    api_key: Optional[str] = None
    enable_streaming: bool = True
    enable_batch_mode: bool = True
    enable_conversation_tracking: bool = True
    enable_memory_profiling: bool = True
    enable_bandwidth_tracking: bool = True
    conversation_length: int = 5  # Number of turns per conversation
    streaming_chunk_size: int = 1  # Tokens per chunk for streaming

    def __post_init__(self):
        if self.test_levels is None:
            self.test_levels = [50, 150]  # As specified in docstring


@dataclass 
class StreamingMetrics:
    """Metrics for streaming and conversational tasks"""
    task_id: str
    conversation_id: str
    turn_number: int
    task_type: str  # streaming, batch, conversation
    timestamp: float
    completion_time: float
    time_to_first_token: float  # TTFT - critical streaming metric
    tokens_per_second: float
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    success: bool
    error_message: Optional[str] = None
    streaming_chunks: int = 0
    chunk_intervals: List[float] = None  # Time between chunks
    conversation_context_length: int = 0
    response_coherence_score: float = 0.0  # 0-1 scale
    bandwidth_usage_bytes: int = 0
    memory_usage_mb: float = 0.0
    queue_wait_time: float = 0.0

    def __post_init__(self):
        if self.chunk_intervals is None:
            self.chunk_intervals = []


@dataclass
class StreamingTestResult:
    """Test results for streaming and conversational scenario"""
    level: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_conversations: int
    completed_conversations: int
    
    # Timing metrics
    avg_completion_time: float
    p95_completion_time: float
    p99_completion_time: float
    min_completion_time: float
    max_completion_time: float
    
    # Streaming specific metrics
    avg_time_to_first_token: float
    p95_time_to_first_token: float
    p99_time_to_first_token: float
    min_time_to_first_token: float
    max_time_to_first_token: float
    
    avg_tokens_per_second: float
    p95_tokens_per_second: float
    p99_tokens_per_second: float
    min_tokens_per_second: float
    max_tokens_per_second: float
    
    # Conversation metrics
    avg_conversation_length: float
    avg_context_retention: float
    avg_response_coherence: float
    
    # Performance metrics
    throughput_requests_per_second: float
    throughput_conversations_per_second: float
    error_rate: float
    avg_bandwidth_per_request: float
    total_bandwidth_usage: int
    
    # Resource usage
    avg_cpu_usage: float
    avg_memory_usage: float
    avg_gpu_usage: float
    avg_gpu_memory: float
    peak_cpu_usage: float
    peak_memory_usage: float
    peak_gpu_usage: float
    peak_gpu_memory: float
    avg_memory_per_request: float
    peak_memory_per_request: float
    
    # Queue metrics
    avg_queue_wait_time: float
    max_queue_wait_time: float
    
    test_duration: float
    total_tokens_generated: int
    
    # Distributions and breakdowns
    completion_time_distribution: Dict[str, float]
    ttft_distribution: Dict[str, float]
    tokens_per_second_distribution: Dict[str, float]
    task_type_distribution: Dict[str, int]
    error_types: Dict[str, int]
    streaming_quality_stats: Dict[str, float]


class StreamingQueueMonitor:
    """Monitor request queue for streaming operations"""
    
    def __init__(self):
        self.queue_depths = []
        self.wait_times = []
        self.streaming_waits = []
        self.batch_waits = []
        self.current_queue_depth = 0
        self.lock = threading.Lock()
    
    def add_to_queue(self, request_type: str = "unknown") -> float:
        """Add request to queue and return entry time"""
        with self.lock:
            self.current_queue_depth += 1
            self.queue_depths.append(self.current_queue_depth)
            return time.time()
    
    def remove_from_queue(self, entry_time: float, request_type: str = "unknown"):
        """Remove request from queue and record wait time"""
        with self.lock:
            self.current_queue_depth = max(0, self.current_queue_depth - 1)
            wait_time = time.time() - entry_time
            self.wait_times.append(wait_time)
            
            if request_type == "streaming":
                self.streaming_waits.append(wait_time)
            elif request_type == "batch":
                self.batch_waits.append(wait_time)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive queue statistics"""
        with self.lock:
            return {
                "avg_queue_depth": statistics.mean(self.queue_depths) if self.queue_depths else 0,
                "max_queue_depth": max(self.queue_depths) if self.queue_depths else 0,
                "avg_wait_time": statistics.mean(self.wait_times) if self.wait_times else 0,
                "max_wait_time": max(self.wait_times) if self.wait_times else 0,
                "avg_streaming_wait": statistics.mean(self.streaming_waits) if self.streaming_waits else 0,
                "avg_batch_wait": statistics.mean(self.batch_waits) if self.batch_waits else 0,
                "total_requests": len(self.wait_times)
            }


class StreamingResourceMonitor:
    """Enhanced resource monitoring for streaming workloads"""
    
    def __init__(self, monitoring_interval: float = 0.5):
        self.monitoring_interval = monitoring_interval
        self.cpu_readings = []
        self.memory_readings = []
        self.gpu_readings = []
        self.gpu_memory_readings = []
        self.network_readings = []
        self.disk_io_readings = []
        self.monitoring = False
        self.monitor_thread = None
        self.start_network_stats = None
        self.start_disk_stats = None

    def start_monitoring(self):
        """Start enhanced resource monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.cpu_readings.clear()
        self.memory_readings.clear()
        self.gpu_readings.clear()
        self.gpu_memory_readings.clear()
        self.network_readings.clear()
        self.disk_io_readings.clear()
        
        # Get initial stats
        try:
            self.start_network_stats = psutil.net_io_counters()
            self.start_disk_stats = psutil.disk_io_counters()
        except Exception:
            self.start_network_stats = None
            self.start_disk_stats = None
        
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Enhanced streaming resource monitoring started")

    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Streaming resource monitoring stopped")

    def _monitor_resources(self):
        """Internal monitoring loop with comprehensive metrics"""
        while self.monitoring:
            try:
                # CPU and Memory
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                self.cpu_readings.append(cpu_percent)
                self.memory_readings.append(memory_percent)

                # Network I/O
                try:
                    net_stats = psutil.net_io_counters()
                    if self.start_network_stats:
                        bytes_sent = net_stats.bytes_sent - self.start_network_stats.bytes_sent
                        bytes_recv = net_stats.bytes_recv - self.start_network_stats.bytes_recv
                        self.network_readings.append(bytes_sent + bytes_recv)
                except Exception:
                    self.network_readings.append(0)

                # Disk I/O
                try:
                    disk_stats = psutil.disk_io_counters()
                    if self.start_disk_stats:
                        read_bytes = disk_stats.read_bytes - self.start_disk_stats.read_bytes
                        write_bytes = disk_stats.write_bytes - self.start_disk_stats.write_bytes
                        self.disk_io_readings.append(read_bytes + write_bytes)
                except Exception:
                    self.disk_io_readings.append(0)

                # GPU monitoring
                if GPU_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]
                            self.gpu_readings.append(gpu.load * 100)
                            self.gpu_memory_readings.append(gpu.memoryUtil * 100)
                        else:
                            self.gpu_readings.append(0)
                            self.gpu_memory_readings.append(0)
                    except Exception as e:
                        logger.warning(f"GPU monitoring failed: {e}")
                        self.gpu_readings.append(0)
                        self.gpu_memory_readings.append(0)
                else:
                    self.gpu_readings.append(0)
                    self.gpu_memory_readings.append(0)

            except Exception as e:
                logger.error(f"Error in streaming resource monitoring: {e}")

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
            "gpu_memory": safe_stats(self.gpu_memory_readings),
            "network_bytes": sum(self.network_readings) if self.network_readings else 0,
            "disk_io_bytes": sum(self.disk_io_readings) if self.disk_io_readings else 0
        }


class ConversationalAgent:
    """LangGraph-inspired conversational agent with streaming capabilities"""
    
    @staticmethod
    def get_conversation_starters() -> List[Tuple[str, str]]:
        """Get conversation starter prompts with expected response types"""
        return [
            ("Tell me about the history of artificial intelligence.", "informational"),
            ("What's the weather like today? Can you help me plan my outfit?", "practical"),
            ("I'm feeling stressed about work. Can you give me some advice?", "supportive"),
            ("Explain quantum computing in simple terms.", "educational"),
            ("What are some good recipes for dinner tonight?", "practical"),
            ("How do I learn a new programming language effectively?", "educational"),
            ("What's your opinion on the future of renewable energy?", "analytical"),
            ("Can you help me write a creative story about space exploration?", "creative"),
            ("I need help organizing my daily schedule. Any tips?", "practical"),
            ("What are the philosophical implications of consciousness?", "philosophical")
        ]
    
    @staticmethod
    def get_follow_up_prompts() -> List[str]:
        """Get follow-up prompts to continue conversations"""
        return [
            "Can you elaborate on that point?",
            "What would you recommend as the next step?",
            "That's interesting. How does this relate to current trends?",
            "Can you give me a specific example?",
            "What are the potential challenges with this approach?",
            "How can I apply this in my daily life?",
            "What are the alternatives to consider?",
            "Can you break this down into simpler steps?",
            "What should I avoid when implementing this?",
            "How has this evolved over time?"
        ]
    
    @staticmethod
    def generate_conversation_context(conversation_history: List[str], turn_number: int) -> str:
        """Generate conversation context for the next turn"""
        if turn_number == 0:
            starter, _ = random.choice(ConversationalAgent.get_conversation_starters())
            return starter
        else:
            if turn_number < 3:  # Early conversation
                return random.choice(ConversationalAgent.get_follow_up_prompts())
            else:  # Later conversation - more context-specific
                context_prompts = [
                    "Thank you for that explanation. One more question:",
                    "Building on what you said earlier:",
                    "I'd like to explore this topic further:",
                    "Can we discuss the practical implications of:",
                    "What would be your final recommendation about:"
                ]
                return random.choice(context_prompts)
    
    @staticmethod
    def evaluate_conversation_coherence(conversation_history: List[str], latest_response: str) -> float:
        """Evaluate how well the response fits the conversation context"""
        if not conversation_history or not latest_response:
            return 0.5
        
        # Simple coherence scoring based on keyword overlap and context awareness
        previous_context = " ".join(conversation_history[-2:]).lower()  # Last 2 exchanges
        response_lower = latest_response.lower()
        
        # Look for context references
        context_indicators = ["as i mentioned", "building on that", "regarding your question", 
                            "following up", "in relation to", "concerning", "about that",
                            "you asked", "previously", "earlier", "as we discussed"]
        
        coherence_score = 0.5  # Base score
        
        # Check for explicit context references
        if any(indicator in response_lower for indicator in context_indicators):
            coherence_score += 0.2
        
        # Check for keyword continuity
        context_words = set(word for word in previous_context.split() if len(word) > 3)
        response_words = set(word for word in response_lower.split() if len(word) > 3)
        
        if context_words and response_words:
            overlap_ratio = len(context_words.intersection(response_words)) / len(context_words)
            coherence_score += overlap_ratio * 0.3
        
        return min(coherence_score, 1.0)
    
    @staticmethod
    def evaluate_response_quality(response: str, response_type: str) -> float:
        """Evaluate the quality of the response based on type"""
        if not response:
            return 0.0
        
        response_lower = response.lower()
        quality_indicators = {
            "informational": ["according to", "research", "studies", "evidence", "facts", "data"],
            "practical": ["step", "first", "then", "next", "how to", "method", "approach"],
            "supportive": ["understand", "feel", "support", "help", "comfort", "reassure"],
            "educational": ["explain", "concept", "definition", "example", "principle"],
            "analytical": ["analysis", "consider", "factors", "perspective", "viewpoint"],
            "creative": ["imagine", "story", "creative", "artistic", "innovative"]
        }
        
        indicators = quality_indicators.get(response_type, [])
        found_indicators = sum(1 for indicator in indicators if indicator in response_lower)
        
        # Base scoring
        indicator_score = min(found_indicators / max(len(indicators), 1), 1.0) if indicators else 0.5
        length_score = min(len(response.split()) / 100, 1.0)  # Normalize by expected length
        
        return (indicator_score * 0.7) + (length_score * 0.3)


class OllamaStreamingClient:
    """Ollama client optimized for streaming and conversational scenarios"""
    
    def __init__(self, config: StreamingTestConfig):
        self.config = config
        self.client = ollama.Client(host=config.base_url)
    
    def send_streaming_request(self, prompt: str, conversation_history: List[str], 
                             task_id: str) -> Tuple[bool, float, float, Optional[str], int, int, int, float, int, List[float]]:
        """
        Send streaming request to Ollama with comprehensive metrics
        Returns: (success, completion_time, ttft, response_text, prompt_tokens, completion_tokens, 
                 total_tokens, memory_usage_mb, chunk_count, chunk_intervals)
        """
        # Start memory profiling
        if self.config.enable_memory_profiling:
            tracemalloc.start()
        
        start_time = time.time()
        first_token_time = None
        response_chunks = []
        chunk_intervals = []
        last_chunk_time = start_time
        chunk_count = 0
        
        try:
            # Build conversation context
            messages = []
            if conversation_history:
                for i, msg in enumerate(conversation_history):
                    role = "user" if i % 2 == 0 else "assistant"
                    messages.append({"role": role, "content": msg})
            
            messages.append({"role": "user", "content": prompt})
            
            logger.debug(f"Sending streaming request for task {task_id}")
            
            # Stream response from Ollama
            response_stream = self.client.chat(
                model=self.config.model_name,
                messages=messages,
                stream=True,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            )
            
            for chunk in response_stream:
                current_time = time.time()
                
                # Record time to first token
                if first_token_time is None:
                    first_token_time = current_time
                
                # Record chunk interval
                if chunk_count > 0:
                    interval = current_time - last_chunk_time
                    chunk_intervals.append(interval)
                
                # Extract content from chunk
                content = chunk.get("message", {}).get("content", "")
                if content:
                    response_chunks.append(content)
                    chunk_count += 1
                
                last_chunk_time = current_time
                
                # Check if done
                if chunk.get("done", False):
                    break
            
            end_time = time.time()
            completion_time = end_time - start_time
            ttft = (first_token_time - start_time) if first_token_time else completion_time
            
            # Combine response chunks
            response_text = "".join(response_chunks)
            
            # Get memory usage
            memory_usage_mb = 0
            if self.config.enable_memory_profiling:
                try:
                    current, peak = tracemalloc.get_traced_memory()
                    memory_usage_mb = peak / 1024 / 1024
                    tracemalloc.stop()
                except Exception:
                    memory_usage_mb = 0
            
            # Estimate token usage
            full_context = " ".join(messages[m]["content"] for m in range(len(messages)))
            prompt_tokens = len(full_context.split()) * 1.3
            completion_tokens = len(response_text.split()) * 1.3 if response_text else 0
            total_tokens = int(prompt_tokens + completion_tokens)
            
            return True, completion_time, ttft, response_text, int(prompt_tokens), int(completion_tokens), total_tokens, memory_usage_mb, chunk_count, chunk_intervals
                
        except Exception as e:
            end_time = time.time()
            completion_time = end_time - start_time
            ttft = completion_time  # Fallback TTFT
            
            if self.config.enable_memory_profiling:
                try:
                    tracemalloc.stop()
                except Exception:
                    pass
            
            logger.error(f"Streaming request failed: {e}")
            return False, completion_time, ttft, str(e), 0, 0, 0, 0, 0, []
    
    def check_health(self) -> bool:
        """Check if Ollama service is healthy"""
        try:
            logger.info("Checking health of Ollama streaming service")
            models = self.client.list()
            logger.info(f"Streaming health check successful, found {len(models.get('models', []))} models")
            return True
        except Exception as e:
            logger.error(f"Streaming health check failed: {e}")
            return False


class StreamingPerformanceTester:
    """Comprehensive performance tester for streaming and conversational scenarios"""
    
    def __init__(self, config: StreamingTestConfig):
        self.config = config
        self.results: List[StreamingTestResult] = []
        self.queue_monitor = StreamingQueueMonitor()
        self.conversation_states: Dict[str, List[str]] = {}  # Track conversation history
    
    def check_service_health(self) -> bool:
        """Check if Ollama service is available"""
        client = OllamaStreamingClient(self.config)
        return client.check_health()
    
    def run_warmup(self) -> None:
        """Run warmup with streaming tasks"""
        logger.info(f"Starting streaming warmup for {self.config.warmup_duration} seconds...")
        
        client = OllamaStreamingClient(self.config)
        warmup_end = time.time() + self.config.warmup_duration
        warmup_count = 0
        
        while time.time() < warmup_end:
            starter, _ = random.choice(ConversationalAgent.get_conversation_starters())
            
            try:
                success, _, _, _, _, _, _, _, _, _ = client.send_streaming_request(starter, [], f"warmup_{warmup_count}")
                warmup_count += 1
                if success:
                    logger.debug(f"Warmup task {warmup_count} completed")
            except Exception as e:
                logger.warning(f"Warmup task failed: {e}")
            
            time.sleep(1)  # Brief delay for warmup
        
        logger.info(f"Streaming warmup completed with {warmup_count} tasks")

    def run_single_streaming_task(self, client: OllamaStreamingClient, task_id: str, 
                                 conversation_id: str, turn_number: int) -> StreamingMetrics:
        """Run a single streaming task with comprehensive metrics"""
        queue_entry_time = self.queue_monitor.add_to_queue("streaming")
        
        # Get conversation history
        conversation_history = self.conversation_states.get(conversation_id, [])
        
        # Generate prompt based on conversation turn
        prompt = ConversationalAgent.generate_conversation_context(conversation_history, turn_number)
        timestamp = time.time()
        
        # Note: response_type could be used for quality evaluation in future enhancements
        
        success, completion_time, ttft, response, prompt_tokens, completion_tokens, total_tokens, memory_usage, chunk_count, chunk_intervals = \
            client.send_streaming_request(prompt, conversation_history, task_id)
        
        # Remove from queue
        self.queue_monitor.remove_from_queue(queue_entry_time, "streaming")
        queue_wait_time = timestamp - queue_entry_time if queue_entry_time < timestamp else 0
        
        # Calculate tokens per second
        tokens_per_second = completion_tokens / completion_time if completion_time > 0 and completion_tokens > 0 else 0
        
        # Update conversation history
        if success and response and self.config.enable_conversation_tracking:
            if conversation_id not in self.conversation_states:
                self.conversation_states[conversation_id] = []
            
            self.conversation_states[conversation_id].append(prompt)
            self.conversation_states[conversation_id].append(response)
            
            # Keep conversation history manageable
            if len(self.conversation_states[conversation_id]) > self.config.conversation_length * 2:
                self.conversation_states[conversation_id] = self.conversation_states[conversation_id][-self.config.conversation_length * 2:]
        
        # Evaluate conversation coherence
        coherence_score = 0.0
        if success and response and conversation_history:
            coherence_score = ConversationalAgent.evaluate_conversation_coherence(conversation_history, response)
        
        # Estimate bandwidth usage (rough calculation)
        bandwidth_usage = len(prompt.encode('utf-8')) + len(response.encode('utf-8')) if response else len(prompt.encode('utf-8'))
        
        return StreamingMetrics(
            task_id=task_id,
            conversation_id=conversation_id,
            turn_number=turn_number,
            task_type="streaming",
            timestamp=timestamp,
            completion_time=completion_time,
            time_to_first_token=ttft,
            tokens_per_second=tokens_per_second,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            success=success,
            error_message=response if not success else None,
            streaming_chunks=chunk_count,
            chunk_intervals=chunk_intervals,
            conversation_context_length=len(conversation_history),
            response_coherence_score=coherence_score,
            bandwidth_usage_bytes=bandwidth_usage,
            memory_usage_mb=memory_usage,
            queue_wait_time=queue_wait_time
        )

    def run_test_level(self, concurrent_users: int) -> StreamingTestResult:
        """Run streaming test for a specific concurrency level"""
        logger.info(f"Starting streaming test with {concurrent_users} concurrent users for {self.config.test_duration} seconds")
        
        # Start enhanced monitoring
        monitor = StreamingResourceMonitor()
        monitor.start_monitoring()
        
        # Initialize metrics collection
        all_metrics: List[StreamingMetrics] = []
        test_start_time = time.time()
        test_end_time = test_start_time + self.config.test_duration
        
        # Clear conversation states for this test
        self.conversation_states.clear()
        
        logger.info("Streaming test metrics initialized")
        
        # Worker function for streaming/conversational tasks
        def streaming_worker(worker_id: int) -> List[StreamingMetrics]:
            logger.info(f"Streaming worker {worker_id} started")
            client = OllamaStreamingClient(self.config)
            worker_metrics = []
            task_count = 0
            
            # Each worker manages multiple conversations
            worker_conversations = {}
            
            while time.time() < test_end_time:
                try:
                    # Decide on conversation ID (create new or continue existing)
                    if len(worker_conversations) < 3 or random.random() < 0.3:  # Start new conversation
                        conversation_id = f"worker_{worker_id}_conv_{len(worker_conversations)}"
                        turn_number = 0
                        worker_conversations[conversation_id] = 0
                    else:  # Continue existing conversation
                        conversation_id = random.choice(list(worker_conversations.keys()))
                        turn_number = worker_conversations[conversation_id]
                        
                        # End conversation if it gets too long
                        if turn_number >= self.config.conversation_length:
                            del worker_conversations[conversation_id]
                            continue
                    
                    task_id = f"worker_{worker_id}_task_{task_count}"
                    metrics = self.run_single_streaming_task(client, task_id, conversation_id, turn_number)
                    worker_metrics.append(metrics)
                    
                    # Update conversation turn count
                    if conversation_id in worker_conversations:
                        worker_conversations[conversation_id] += 1
                    
                    task_count += 1
                    
                    if task_count % 10 == 0:
                        logger.debug(f"Streaming worker {worker_id}: {task_count} tasks completed")
                    
                except Exception as e:
                    logger.error(f"Streaming worker {worker_id} error: {e}")
                    worker_metrics.append(StreamingMetrics(
                        task_id=f"worker_{worker_id}_error_{task_count}",
                        conversation_id="error",
                        turn_number=0,
                        task_type="error",
                        timestamp=time.time(),
                        completion_time=0,
                        time_to_first_token=0,
                        tokens_per_second=0,
                        total_tokens=0,
                        prompt_tokens=0,
                        completion_tokens=0,
                        success=False,
                        error_message=str(e)
                    ))
                
                # Brief delay for realistic conversation pacing
                time.sleep(0.1)
            
            logger.info(f"Streaming worker {worker_id} completed with {task_count} tasks")
            return worker_metrics
        
        # Run workers in parallel
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(streaming_worker, i) for i in range(concurrent_users)]
            
            for future in as_completed(futures):
                try:
                    worker_metrics = future.result()
                    all_metrics.extend(worker_metrics)
                except Exception as e:
                    logger.error(f"Streaming worker failed: {e}")
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        actual_test_duration = time.time() - test_start_time
        
        logger.info(f"Streaming test level {concurrent_users} completed. Processing {len(all_metrics)} metrics...")
        
        return self._calculate_streaming_test_result(concurrent_users, all_metrics, monitor, actual_test_duration)

    def _calculate_streaming_test_result(self, level: int, metrics: List[StreamingMetrics], 
                                       monitor: StreamingResourceMonitor, test_duration: float) -> StreamingTestResult:
        """Calculate comprehensive test results for streaming scenarios"""
        
        successful_metrics = [m for m in metrics if m.success]
        failed_metrics = [m for m in metrics if not m.success]
        
        total_requests = len(metrics)
        successful_requests = len(successful_metrics)
        failed_requests = len(failed_metrics)
        
        # Count conversations
        conversation_ids = set(m.conversation_id for m in metrics if m.conversation_id != "error")
        total_conversations = len(conversation_ids)
        completed_conversations = len([cid for cid in conversation_ids 
                                     if max([m.turn_number for m in metrics if m.conversation_id == cid], default=0) >= 2])
        
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
                "p99": p99_completion_time
            }
        else:
            avg_completion_time = p95_completion_time = p99_completion_time = 0
            min_completion_time = max_completion_time = 0
            completion_time_distribution = {}
        
        # Time to First Token (TTFT) calculations
        if successful_metrics:
            ttft_times = [m.time_to_first_token for m in successful_metrics if m.time_to_first_token > 0]
            if ttft_times:
                avg_ttft = statistics.mean(ttft_times)
                p95_ttft = np.percentile(ttft_times, 95)
                p99_ttft = np.percentile(ttft_times, 99)
                min_ttft = min(ttft_times)
                max_ttft = max(ttft_times)
                
                ttft_distribution = {
                    "p50": np.percentile(ttft_times, 50),
                    "p75": np.percentile(ttft_times, 75),
                    "p90": np.percentile(ttft_times, 90),
                    "p95": p95_ttft,
                    "p99": p99_ttft
                }
            else:
                avg_ttft = p95_ttft = p99_ttft = min_ttft = max_ttft = 0
                ttft_distribution = {}
        else:
            avg_ttft = p95_ttft = p99_ttft = min_ttft = max_ttft = 0
            ttft_distribution = {}
        
        # Tokens per second calculations
        if successful_metrics:
            tps_values = [m.tokens_per_second for m in successful_metrics if m.tokens_per_second > 0]
            if tps_values:
                avg_tps = statistics.mean(tps_values)
                p95_tps = np.percentile(tps_values, 95)
                p99_tps = np.percentile(tps_values, 99)
                min_tps = min(tps_values)
                max_tps = max(tps_values)
                
                tps_distribution = {
                    "p50": np.percentile(tps_values, 50),
                    "p75": np.percentile(tps_values, 75),
                    "p90": np.percentile(tps_values, 90),
                    "p95": p95_tps,
                    "p99": p99_tps
                }
            else:
                avg_tps = p95_tps = p99_tps = min_tps = max_tps = 0
                tps_distribution = {}
        else:
            avg_tps = p95_tps = p99_tps = min_tps = max_tps = 0
            tps_distribution = {}
        
        # Conversation metrics
        if successful_metrics:
            conversation_lengths = [m.conversation_context_length for m in successful_metrics]
            coherence_scores = [m.response_coherence_score for m in successful_metrics if m.response_coherence_score > 0]
            
            avg_conversation_length = statistics.mean(conversation_lengths) if conversation_lengths else 0
            avg_response_coherence = statistics.mean(coherence_scores) if coherence_scores else 0
            avg_context_retention = min(avg_conversation_length / self.config.conversation_length, 1.0)
        else:
            avg_conversation_length = avg_response_coherence = avg_context_retention = 0
        
        # Performance calculations
        throughput_rps = successful_requests / test_duration if test_duration > 0 else 0
        throughput_cps = completed_conversations / test_duration if test_duration > 0 else 0
        error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0
        
        # Bandwidth calculations
        bandwidth_usages = [m.bandwidth_usage_bytes for m in successful_metrics]
        avg_bandwidth_per_request = statistics.mean(bandwidth_usages) if bandwidth_usages else 0
        total_bandwidth_usage = sum(bandwidth_usages) if bandwidth_usages else 0
        
        # Memory calculations
        memory_usages = [m.memory_usage_mb for m in successful_metrics if m.memory_usage_mb > 0]
        avg_memory_per_request = statistics.mean(memory_usages) if memory_usages else 0
        peak_memory_per_request = max(memory_usages) if memory_usages else 0
        
        # Queue statistics
        queue_stats = self.queue_monitor.get_statistics()
        avg_queue_wait_time = queue_stats["avg_wait_time"]
        max_queue_wait_time = queue_stats["max_wait_time"]
        
        # Token statistics
        total_tokens = sum(m.total_tokens for m in successful_metrics)
        
        # Task type distribution
        task_type_distribution = {}
        for m in metrics:
            task_type_distribution[m.task_type] = task_type_distribution.get(m.task_type, 0) + 1
        
        # Error types
        error_types = {}
        for m in failed_metrics:
            error_msg = m.error_message or "Unknown error"
            error_types[error_msg] = error_types.get(error_msg, 0) + 1
        
        # Streaming quality statistics
        streaming_quality_stats = {}
        if successful_metrics:
            chunk_counts = [m.streaming_chunks for m in successful_metrics if m.streaming_chunks > 0]
            streaming_quality_stats = {
                "avg_chunks": statistics.mean(chunk_counts) if chunk_counts else 0,
                "max_chunks": max(chunk_counts) if chunk_counts else 0,
                "total_streaming_requests": len(successful_metrics)
            }
        
        # Resource usage statistics
        resource_stats = monitor.get_statistics()
        
        return StreamingTestResult(
            level=level,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_conversations=total_conversations,
            completed_conversations=completed_conversations,
            
            avg_completion_time=avg_completion_time,
            p95_completion_time=p95_completion_time,
            p99_completion_time=p99_completion_time,
            min_completion_time=min_completion_time,
            max_completion_time=max_completion_time,
            
            avg_time_to_first_token=avg_ttft,
            p95_time_to_first_token=p95_ttft,
            p99_time_to_first_token=p99_ttft,
            min_time_to_first_token=min_ttft,
            max_time_to_first_token=max_ttft,
            
            avg_tokens_per_second=avg_tps,
            p95_tokens_per_second=p95_tps,
            p99_tokens_per_second=p99_tps,
            min_tokens_per_second=min_tps,
            max_tokens_per_second=max_tps,
            
            avg_conversation_length=avg_conversation_length,
            avg_context_retention=avg_context_retention,
            avg_response_coherence=avg_response_coherence,
            
            throughput_requests_per_second=throughput_rps,
            throughput_conversations_per_second=throughput_cps,
            error_rate=error_rate,
            avg_bandwidth_per_request=avg_bandwidth_per_request,
            total_bandwidth_usage=total_bandwidth_usage,
            
            avg_cpu_usage=resource_stats["cpu"]["avg"],
            avg_memory_usage=resource_stats["memory"]["avg"],
            avg_gpu_usage=resource_stats["gpu"]["avg"],
            avg_gpu_memory=resource_stats["gpu_memory"]["avg"],
            peak_cpu_usage=resource_stats["cpu"]["max"],
            peak_memory_usage=resource_stats["memory"]["max"],
            peak_gpu_usage=resource_stats["gpu"]["max"],
            peak_gpu_memory=resource_stats["gpu_memory"]["max"],
            avg_memory_per_request=avg_memory_per_request,
            peak_memory_per_request=peak_memory_per_request,
            
            avg_queue_wait_time=avg_queue_wait_time,
            max_queue_wait_time=max_queue_wait_time,
            
            test_duration=test_duration,
            total_tokens_generated=total_tokens,
            
            completion_time_distribution=completion_time_distribution,
            ttft_distribution=ttft_distribution,
            tokens_per_second_distribution=tps_distribution,
            task_type_distribution=task_type_distribution,
            error_types=error_types,
            streaming_quality_stats=streaming_quality_stats
        )

    def run_all_tests(self) -> List[StreamingTestResult]:
        """Run all streaming test levels"""
        logger.info("=" * 80)
        logger.info("STARTING OLLAMA STREAMING & CONVERSATIONAL PERFORMANCE TESTS")
        logger.info("=" * 80)
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Deployment: {self.config.deployment_type}")
        logger.info(f"Base URL: {self.config.base_url}")
        logger.info(f"Test levels: {self.config.test_levels}")
        logger.info(f"Test duration per level: {self.config.test_duration} seconds")
        logger.info(f"Streaming enabled: {self.config.enable_streaming}")
        logger.info(f"Conversation tracking: {self.config.enable_conversation_tracking}")
        
        # Check service health
        if not self.check_service_health():
            logger.error("Ollama service is not available. Please check the service.")
            return []
        
        logger.info("✓ Ollama streaming service is healthy")
        
        # Run warmup
        self.run_warmup()
        
        results = []
        
        for i, level in enumerate(self.config.test_levels):
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"STREAMING TEST {i+1}/{len(self.config.test_levels)}: {level} CONCURRENT USERS")
                logger.info(f"{'='*60}")
                
                result = self.run_test_level(level)
                results.append(result)
                
                logger.info(f"Streaming test level {level} completed:")
                logger.info(f"  - Total requests: {result.total_requests}")
                logger.info(f"  - Successful: {result.successful_requests}")
                logger.info(f"  - Failed: {result.failed_requests}")
                logger.info(f"  - Conversations: {result.total_conversations} ({result.completed_conversations} completed)")
                logger.info(f"  - Throughput: {result.throughput_requests_per_second:.2f} req/s")
                logger.info(f"  - Avg TTFT: {result.avg_time_to_first_token:.3f}s")
                logger.info(f"  - Avg TPS: {result.avg_tokens_per_second:.1f} tokens/s")
                logger.info(f"  - Error rate: {result.error_rate:.2f}%")
                
                # Cooldown between tests
                if i < len(self.config.test_levels) - 1:
                    logger.info(f"Cooldown for {self.config.cooldown_duration} seconds...")
                    time.sleep(self.config.cooldown_duration)
                
            except Exception as e:
                logger.error(f"Streaming test level {level} failed: {e}")
                continue
        
        logger.info("\n" + "="*80)
        logger.info("ALL STREAMING TESTS COMPLETED")
        logger.info("="*80)
        
        return results

    def generate_report(self, results: List[StreamingTestResult]) -> str:
        """Generate comprehensive streaming test report"""
        report = []
        
        # Header
        report.append("=" * 100)
        report.append("OLLAMA STREAMING & CONVERSATIONAL PERFORMANCE TEST REPORT - QWEN3-4B")
        report.append("=" * 100)
        report.append(f"Model: {self.config.model_name}")
        report.append(f"Deployment: {self.config.deployment_type}")
        report.append(f"Base URL: {self.config.base_url}")
        report.append(f"Test Duration per Level: {self.config.test_duration} seconds ({self.config.test_duration//60} minutes)")
        report.append(f"Max Tokens: {self.config.max_tokens}")
        report.append(f"Temperature: {self.config.temperature}")
        report.append(f"Streaming Enabled: {self.config.enable_streaming}")
        report.append(f"Conversation Length: {self.config.conversation_length} turns")
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
        total_conversations = sum(r.total_conversations for r in results)
        avg_ttft = statistics.mean([r.avg_time_to_first_token for r in results if r.avg_time_to_first_token > 0])
        max_throughput = max(r.throughput_requests_per_second for r in results)
        
        report.append(f"Total Requests Processed: {total_requests:,}")
        report.append(f"Overall Success Rate: {(total_successful/total_requests*100):.2f}%" if total_requests > 0 else "N/A")
        report.append(f"Total Conversations: {total_conversations}")
        report.append(f"Average Time to First Token: {avg_ttft:.3f}s" if avg_ttft > 0 else "N/A")
        report.append(f"Maximum Throughput: {max_throughput:.2f} requests/s")
        report.append("")
        
        # Performance Summary Table
        report.append("STREAMING PERFORMANCE SUMMARY")
        report.append("-" * 160)
        report.append(f"{'Level':<8} {'Requests':<9} {'Success':<9} {'Failed':<8} {'Convs':<7} {'RPS':<7} "
                      f"{'TTFT(s)':<9} {'TPS':<7} {'Bandwidth':<11} {'Coherence':<10} {'Memory':<8}")
        report.append("-" * 160)
        
        for result in results:
            report.append(f"{result.level:<8} {result.total_requests:<9} {result.successful_requests:<9} "
                          f"{result.failed_requests:<8} {result.total_conversations:<7} "
                          f"{result.throughput_requests_per_second:<7.1f} {result.avg_time_to_first_token:<9.3f} "
                          f"{result.avg_tokens_per_second:<7.1f} {result.avg_bandwidth_per_request/1024:<11.1f} "
                          f"{result.avg_response_coherence*100:<10.1f} {result.avg_memory_per_request:<8.1f}")
        
        report.append("")
        
        # Detailed Metrics
        report.append("DETAILED STREAMING METRICS")
        report.append("-" * 80)
        for result in results:
            report.append(f"Level {result.level}:")
            report.append(f"  - P95 TTFT: {result.p95_time_to_first_token:.3f}s")
            report.append(f"  - P99 TTFT: {result.p99_time_to_first_token:.3f}s")
            report.append(f"  - P95 TPS: {result.p95_tokens_per_second:.1f}")
            report.append(f"  - P99 TPS: {result.p99_tokens_per_second:.1f}")
            report.append(f"  - Context Retention: {result.avg_context_retention*100:.1f}%")
            report.append(f"  - Total Bandwidth: {result.total_bandwidth_usage/1024/1024:.1f} MB")
            report.append("")
        
        # Resource Usage
        report.append("RESOURCE USAGE ANALYSIS")
        report.append("-" * 80)
        report.append(f"{'Level':<8} {'CPU%':<8} {'Memory%':<10} {'GPU%':<8} {'GPU Mem%':<10} {'Network MB':<12}")
        report.append("-" * 80)
        
        for result in results:
            network_mb = getattr(result, 'total_bandwidth_usage', 0) / 1024 / 1024
            report.append(f"{result.level:<8} {result.avg_cpu_usage:<8.1f} {result.avg_memory_usage:<10.1f} "
                          f"{result.avg_gpu_usage:<8.1f} {result.avg_gpu_memory:<10.1f} {network_mb:<12.1f}")
        
        report.append("")
        
        # Recommendations
        report.append("OLLAMA STREAMING SPECIFIC RECOMMENDATIONS")
        report.append("-" * 50)
        
        if results:
            best_ttft_result = min(results, key=lambda r: r.avg_time_to_first_token if r.avg_time_to_first_token > 0 else float('inf'))
            best_tps_result = max(results, key=lambda r: r.avg_tokens_per_second)
            best_coherence_result = max(results, key=lambda r: r.avg_response_coherence)
            
            report.append(f"• Best TTFT Performance: {best_ttft_result.level} concurrent users "
                          f"({best_ttft_result.avg_time_to_first_token:.3f}s)")
            report.append(f"• Best Token Throughput: {best_tps_result.level} concurrent users "
                          f"({best_tps_result.avg_tokens_per_second:.1f} tokens/s)")
            report.append(f"• Best Conversation Coherence: {best_coherence_result.level} concurrent users "
                          f"({best_coherence_result.avg_response_coherence*100:.1f}%)")
            
            # Performance analysis
            if any(r.avg_time_to_first_token > 2.0 for r in results):
                report.append("• High TTFT detected - consider optimizing model loading or increasing OLLAMA_NUM_PARALLEL")
            
            if any(r.avg_tokens_per_second < 10 for r in results):
                report.append("• Low token generation speed - consider GPU optimization or model quantization")
            
            if any(r.avg_response_coherence < 0.6 for r in results):
                report.append("• Low conversation coherence - consider adjusting temperature or context window")
        
        report.append("• For streaming optimization, consider:")
        report.append("  - Enabling OLLAMA_FLASH_ATTENTION for faster token generation")
        report.append("  - Setting OLLAMA_NUM_PARALLEL to match your CPU cores")
        report.append("  - Using OLLAMA_MAX_LOADED_MODELS=1 for consistent streaming performance")
        report.append("  - Implementing client-side chunking for better user experience")
        report.append("  - Consider WebSocket connections for real-time streaming")
        
        report.append("")
        report.append("Streaming and conversational test completed successfully!")
        report.append("=" * 100)
        
        return "\n".join(report)

    def save_results(self, results: List[StreamingTestResult], output_dir: str = "results/scenario-3"):
        """Save streaming test results"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save text report
        report = self.generate_report(results)
        report_file = os.path.join(output_dir, f"ollama_streaming_report_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Streaming report saved to {report_file}")
        
        # Save JSON data
        json_file = os.path.join(output_dir, f"ollama_streaming_data_{timestamp}.json")
        json_data = {
            "config": asdict(self.config),
            "timestamp": datetime.now().isoformat(),
            "results": [asdict(result) for result in results]
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Streaming JSON data saved to {json_file}")
        
        return report_file, json_file


def main():
    """Main execution function for streaming tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ollama Streaming & Conversational Performance Test - Qwen3-4B")
    parser.add_argument("--model", default="qwen3:4b", help="Model name")
    parser.add_argument("--url", default="http://localhost:11435", help="Ollama base URL")
    parser.add_argument("--levels", nargs="+", type=int, default=[50, 150],
                        help="Concurrency levels for streaming tests")
    parser.add_argument("--duration", type=int, default=900,
                        help="Test duration per level in seconds (default: 900 = 15 minutes)")
    parser.add_argument("--warmup", type=int, default=30,
                        help="Warmup duration in seconds")
    parser.add_argument("--output-dir", default="results/scenario-3",
                        help="Output directory for results")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Maximum tokens per response for streaming")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for conversation (higher = more creative)")
    parser.add_argument("--conversation-length", type=int, default=5,
                        help="Number of turns per conversation")
    parser.add_argument("--disable-streaming", action="store_true",
                        help="Disable streaming mode (batch only)")
    parser.add_argument("--disable-conversation-tracking", action="store_true",
                        help="Disable conversation context tracking")
    parser.add_argument("--disable-memory-profiling", action="store_true",
                        help="Disable memory profiling")
    
    args = parser.parse_args()
    
    # Create test configuration
    config = StreamingTestConfig(
        model_name=args.model,
        base_url=args.url,
        test_levels=args.levels,
        test_duration=args.duration,
        warmup_duration=args.warmup,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        conversation_length=args.conversation_length,
        enable_streaming=not args.disable_streaming,
        enable_conversation_tracking=not args.disable_conversation_tracking,
        enable_memory_profiling=not args.disable_memory_profiling
    )
    
    # Log configuration
    logger.info("Ollama Streaming Test Configuration:")
    for key, value in asdict(config).items():
        logger.info(f"  {key}: {value}")
    
    # Create and run tester
    tester = StreamingPerformanceTester(config)
    
    try:
        results = tester.run_all_tests()
        
        if results:
            # Save results
            report_file, json_file = tester.save_results(results, args.output_dir)
            
            # Print summary
            print("\n" + "="*80)
            print("OLLAMA STREAMING TEST SUMMARY")
            print("="*80)
            print("Reports saved to:")
            print(f"  - Text report: {report_file}")
            print(f"  - JSON data: {json_file}")
            print(f"Total test levels completed: {len(results)}")
            
            # Print key metrics
            if results:
                total_requests = sum(r.total_requests for r in results)
                total_successful = sum(r.successful_requests for r in results)
                avg_ttft = statistics.mean([r.avg_time_to_first_token for r in results if r.avg_time_to_first_token > 0])
                max_throughput = max(r.throughput_requests_per_second for r in results)
                
                print(f"Total requests processed: {total_requests:,}")
                print(f"Overall success rate: {(total_successful/total_requests*100):.2f}%")
                print(f"Average TTFT: {avg_ttft:.3f}s")
                print(f"Maximum throughput: {max_throughput:.2f} req/s")
            
            print("="*80)
        else:
            logger.error("No streaming test results generated")
            
    except KeyboardInterrupt:
        logger.info("Streaming test interrupted by user")
    except Exception as e:
        logger.error(f"Streaming test failed: {e}")
        raise


if __name__ == "__main__":
    main()
