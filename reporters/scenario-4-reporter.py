#!/usr/bin/env python3
"""
Scenario-4 HTML Report Generator
vLLM stress testing ve load scalability raporunu HTML formatında oluşturur.
Ollama bu senaryoda başarısız olduğu için sadece vLLM sonuçları analiz edilir.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

class Scenario4Reporter:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.vllm_data = None
        
    def load_data(self):
        """JSON verilerini yükler"""
        # VLLM verilerini yükle
        vllm_files = list(self.results_dir.glob("vllm_stress_test_data_*.json"))
        if not vllm_files:
            raise ValueError("vLLM stress test veri dosyası bulunamadı!")
            
        with open(vllm_files[0], 'r') as f:
            self.vllm_data = json.load(f)
    
    def analyze_stress_test_results(self):
        """Stress test sonuçlarını analiz eder"""
        results = self.vllm_data['results']
        config = self.vllm_data['config']
        
        # Gradual load results
        gradual_results = results.get('gradual_load_results', [])
        
        # Endurance test results  
        endurance_results = results.get('endurance_results', {})
        
        # Spike test results
        spike_results = results.get('spike_test_results', [])
        
        # Performance degradation
        degradation = results.get('performance_degradation', {})
        
        # Resource usage
        resource_usage = results.get('resource_usage_analysis', {})
        
        # Memory leak analysis
        memory_analysis = results.get('memory_leak_analysis', {})
        
        # Breaking point analysis
        breaking_point = results.get('breaking_point_analysis', {})
        
        return {
            'config': config,
            'gradual_results': gradual_results,
            'endurance_results': endurance_results,
            'spike_results': spike_results,
            'degradation': degradation,
            'resource_usage': resource_usage,
            'memory_analysis': memory_analysis,
            'breaking_point': breaking_point
        }
    
    def generate_chart_data(self):
        """Chart.js için stress test verilerini hazırlar"""
        analysis = self.analyze_stress_test_results()
        gradual_results = analysis['gradual_results']
        
        # Gradual load test chart data
        load_levels = []
        latencies = []
        throughputs = []
        error_rates = []
        success_rates = []
        token_speeds = []
        p95_latencies = []
        
        for result in gradual_results:
            load_levels.append(result['load_level'])
            latencies.append(result['avg_completion_time'])
            throughputs.append(result['throughput_requests_per_second'])
            error_rates.append(result['error_rate'])
            success_rates.append(100 - result['error_rate'])
            token_speeds.append(result['avg_tokens_per_second'])
            p95_latencies.append(result['p95_completion_time'])
        
        # Spike test chart data
        spike_data = analysis['spike_results']
        spike_labels = []
        spike_latencies = []
        spike_throughputs = []
        spike_error_rates = []
        
        for spike in spike_data:
            spike_labels.append(f"Spike {spike['peak_load']}")
            spike_latencies.append(spike['avg_latency_during_spike'])
            spike_throughputs.append(spike['throughput_during_spike'])
            spike_error_rates.append(spike['error_rate_during_spike'])
        
        # Memory usage over time (if available)
        memory_data = analysis['memory_analysis'].get('memory_samples', [])
        memory_timestamps = []
        memory_usage = []
        
        for sample in memory_data[:20]:  # İlk 20 sample'ı al
            memory_timestamps.append(sample.get('timestamp', ''))
            memory_usage.append(sample.get('memory_mb', 0))
        
        return {
            'load_levels': load_levels,
            'latencies': latencies,
            'throughputs': throughputs,
            'error_rates': error_rates,
            'success_rates': success_rates,
            'token_speeds': token_speeds,
            'p95_latencies': p95_latencies,
            'spike_labels': spike_labels,
            'spike_latencies': spike_latencies,
            'spike_throughputs': spike_throughputs,
            'spike_error_rates': spike_error_rates,
            'memory_timestamps': memory_timestamps,
            'memory_usage': memory_usage
        }
    
    def calculate_summary_metrics(self):
        """Özet metrikleri hesaplar"""
        analysis = self.analyze_stress_test_results()
        
        # Gradual load summary
        gradual_results = analysis['gradual_results']
        total_requests = sum(r['total_requests'] for r in gradual_results)
        total_successful = sum(r['successful_requests'] for r in gradual_results)
        total_failed = sum(r['failed_requests'] for r in gradual_results)
        
        avg_latency = sum(r['avg_completion_time'] for r in gradual_results) / len(gradual_results) if gradual_results else 0
        avg_throughput = sum(r['throughput_requests_per_second'] for r in gradual_results) / len(gradual_results) if gradual_results else 0
        overall_error_rate = (total_failed / total_requests * 100) if total_requests > 0 else 0
        
        # Breaking point info
        breaking_point = analysis['breaking_point']
        bp_load = breaking_point.get('breaking_point_load', 'Not found')
        bp_reason = breaking_point.get('breaking_point_reason', 'Unknown')
        max_stable_load = breaking_point.get('max_stable_load', 'Not determined')
        
        # Endurance test summary
        endurance = analysis['endurance_results']
        endurance_duration = endurance.get('duration_hours', 0)
        endurance_requests = endurance.get('total_requests', 0)
        endurance_success_rate = endurance.get('success_rate', 0)
        endurance_avg_latency = endurance.get('avg_latency', 0)
        
        # Memory analysis
        memory_analysis = analysis['memory_analysis']
        memory_growth_rate = memory_analysis.get('growth_rate_mb_per_hour', 0)
        memory_trend = memory_analysis.get('trend', 'unknown')
        
        # Peak performance
        peak_throughput = max((r['throughput_requests_per_second'] for r in gradual_results)) if gradual_results else 0
        best_latency = min((r['avg_completion_time'] for r in gradual_results)) if gradual_results else float('inf')
        
        return {
            'total_requests': total_requests,
            'total_successful': total_successful,
            'total_failed': total_failed,
            'avg_latency': avg_latency,
            'avg_throughput': avg_throughput,
            'overall_error_rate': overall_error_rate,
            'breaking_point_load': bp_load,
            'breaking_point_reason': bp_reason,
            'max_stable_load': max_stable_load,
            'peak_throughput': peak_throughput,
            'best_latency': best_latency,
            'endurance_duration': endurance_duration,
            'endurance_requests': endurance_requests,
            'endurance_success_rate': endurance_success_rate,
            'endurance_avg_latency': endurance_avg_latency,
            'memory_growth_rate': memory_growth_rate,
            'memory_trend': memory_trend
        }
    
    def generate_recommendations(self):
        """Stress test sonuçlarına göre öneriler oluşturur"""
        analysis = self.analyze_stress_test_results()
        metrics = self.calculate_summary_metrics()
        
        recommendations = []
        
        # Breaking point önerileri
        if metrics['breaking_point_load'] != 'Not found':
            if isinstance(metrics['breaking_point_load'], (int, float)):
                safe_load = int(metrics['breaking_point_load'] * 0.7)
                recommendations.append(f"Production ortamında maksimum {safe_load} concurrent user kullanın (breaking point'in %70'i)")
        
        # Latency önerileri
        if metrics['avg_latency'] > 15:
            recommendations.append("Yüksek latency - --max-num-seqs parametresini artırmayı deneyin")
            recommendations.append("GPU memory optimizasyonu için --enable-chunked-prefill kullanın")
        
        # Error rate önerileri
        if metrics['overall_error_rate'] > 5:
            recommendations.append("Yüksek hata oranı - timeout değerlerini artırın")
            recommendations.append("Connection pool boyutunu artırmayı değerlendirin")
        
        # Memory önerileri
        if metrics['memory_growth_rate'] > 100:  # 100 MB/hour
            recommendations.append("Yüksek memory growth - memory leak kontrolü yapın")
            recommendations.append("--max-model-len parametresini optimize edin")
        
        # Throughput önerileri
        if metrics['peak_throughput'] < 5:
            recommendations.append("Düşük throughput - --tensor-parallel-size kullanarak model paralelliği artırın")
            recommendations.append("--enable-prefix-caching ile performansı artırın")
        
        # Endurance önerileri
        if metrics['endurance_success_rate'] < 95:
            recommendations.append("Endurance test başarı oranı düşük - sistem kararlılığını kontrol edin")
        
        return recommendations

    def generate_html_report(self):
        """HTML raporu oluşturur"""
        chart_data = self.generate_chart_data()
        metrics = self.calculate_summary_metrics()
        recommendations = self.generate_recommendations()
        analysis = self.analyze_stress_test_results()
        
        # Ollama durumu için açıklama
        ollama_failure_note = """
        <div class="ollama-failure-note">
            <h3>⚠️ Ollama Test Durumu</h3>
            <p><strong>Ollama bu stress test senaryosunda başarısız olmuştur.</strong></p>
            <p>Bu nedenle sadece vLLM sonuçları analiz edilmiştir. Karşılaştırmalı analiz mümkün olmamıştır.</p>
        </div>
        """
        
        html_template = f"""
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scenario-4: vLLM Stress Testing & Load Scalability Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 50%, #ff9ff3 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 40px;
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }}
        
        .header h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        }}
        
        .header .subtitle {{
            color: #7f8c8d;
            font-size: 1.2em;
            margin-bottom: 20px;
        }}
        
        .test-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .info-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            border-left: 4px solid #e74c3c;
        }}
        
        .info-card h3 {{
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        
        .ollama-failure-note {{
            background: #fff3cd;
            border: 2px solid #ffc107;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
        }}
        
        .ollama-failure-note h3 {{
            color: #856404;
            margin-bottom: 15px;
        }}
        
        .performance-overview {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }}
        
        .metric-card {{
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
        }}
        
        .metric-card h3 {{
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.3em;
        }}
        
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #27ae60;
            margin: 15px 0;
        }}
        
        .metric-unit {{
            font-size: 0.8em;
            color: #7f8c8d;
        }}
        
        .breaking-point {{
            color: #e74c3c !important;
        }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }}
        
        .chart-container {{
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }}
        
        .chart-container h3 {{
            text-align: center;
            margin-bottom: 20px;
            color: #2c3e50;
            font-size: 1.3em;
        }}
        
        .analysis-section {{
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
            margin-bottom: 30px;
        }}
        
        .analysis-section h2 {{
            color: #2c3e50;
            margin-bottom: 20px;
            border-bottom: 3px solid #e74c3c;
            padding-bottom: 10px;
        }}
        
        .stress-summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .summary-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-top: 4px solid #e74c3c;
        }}
        
        .summary-card h4 {{
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        
        .recommendations {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }}
        
        .recommendations h3 {{
            color: #8b7355;
            margin-bottom: 15px;
        }}
        
        .recommendations ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        
        .recommendations li {{
            background: #fff;
            margin: 10px 0;
            padding: 10px 15px;
            border-radius: 5px;
            border-left: 4px solid #f39c12;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            color: #7f8c8d;
        }}
        
        .status-good {{ color: #27ae60; }}
        .status-warning {{ color: #f39c12; }}
        .status-critical {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔥 Scenario-4: vLLM Stress Testing Analysis</h1>
            <div class="subtitle">Load Scalability & Performance Under Pressure</div>
            
            {ollama_failure_note}
            
            <div class="test-info">
                <div class="info-card">
                    <h3>Model</h3>
                    <p>{analysis['config'].get('model_name', 'Unknown')}</p>
                </div>
                <div class="info-card">
                    <h3>Test Type</h3>
                    <p>Stress Testing</p>
                </div>
                <div class="info-card">
                    <h3>Max Tokens</h3>
                    <p>{analysis['config'].get('max_tokens', 256)}</p>
                </div>
                <div class="info-card">
                    <h3>Temperature</h3>
                    <p>{analysis['config'].get('temperature', 0.3)}</p>
                </div>
                <div class="info-card">
                    <h3>Breaking Point</h3>
                    <p class="breaking-point">{metrics['breaking_point_load']} users</p>
                </div>
                <div class="info-card">
                    <h3>Generated</h3>
                    <p>{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                </div>
            </div>
        </div>
        
        <div class="performance-overview">
            <div class="metric-card">
                <h3>🎯 Total Requests Processed</h3>
                <div class="metric-value">{metrics['total_requests']:,}</div>
                <p class="metric-unit">requests</p>
                <p style="font-size: 0.9em; color: #7f8c8d; margin-top: 10px;">
                    Success Rate: {((metrics['total_successful']/metrics['total_requests'])*100):.1f}%
                </p>
            </div>
            
            <div class="metric-card">
                <h3>⚡ Average Response Time</h3>
                <div class="metric-value">{metrics['avg_latency']:.2f}s</div>
                <p class="metric-unit">seconds</p>
                <p style="font-size: 0.9em; color: #7f8c8d; margin-top: 10px;">
                    Best: {metrics['best_latency']:.2f}s
                </p>
            </div>
            
            <div class="metric-card">
                <h3>📊 Peak Throughput</h3>
                <div class="metric-value">{metrics['peak_throughput']:.2f}</div>
                <p class="metric-unit">requests/second</p>
                <p style="font-size: 0.9em; color: #7f8c8d; margin-top: 10px;">
                    Average: {metrics['avg_throughput']:.2f} req/s
                </p>
            </div>
            
            <div class="metric-card">
                <h3>🔥 Breaking Point</h3>
                <div class="metric-value breaking-point">{metrics['breaking_point_load']}</div>
                <p class="metric-unit">concurrent users</p>
                <p style="font-size: 0.9em; color: #e74c3c; margin-top: 10px;">
                    {metrics['breaking_point_reason'][:50]}...
                </p>
            </div>
            
            <div class="metric-card">
                <h3>⏱️ Endurance Test</h3>
                <div class="metric-value">{metrics['endurance_duration']:.1f}h</div>
                <p class="metric-unit">duration</p>
                <p style="font-size: 0.9em; color: #7f8c8d; margin-top: 10px;">
                    {metrics['endurance_requests']:,} requests - {metrics['endurance_success_rate']:.1f}% success
                </p>
            </div>
            
            <div class="metric-card">
                <h3>🧠 Memory Growth</h3>
                <div class="metric-value {'status-warning' if metrics['memory_growth_rate'] > 100 else 'status-good'}">{metrics['memory_growth_rate']:.1f}</div>
                <p class="metric-unit">MB/hour</p>
                <p style="font-size: 0.9em; color: #7f8c8d; margin-top: 10px;">
                    Trend: {metrics['memory_trend']}
                </p>
            </div>
        </div>

        <div class="stress-summary">
            <div class="summary-card">
                <h4>📈 Gradual Load Test</h4>
                <p><strong>Load Levels:</strong> {len(analysis['gradual_results'])} levels tested</p>
                <p><strong>Max Load:</strong> {max((r['load_level'] for r in analysis['gradual_results'])) if analysis['gradual_results'] else 0} users</p>
                <p><strong>Total Duration:</strong> {(sum(r['duration'] for r in analysis['gradual_results'])/60) if analysis['gradual_results'] else 0.0:.1f} minutes</p>
            </div>
            
            <div class="summary-card">
                <h4>⚡ Spike Test Results</h4>
                <p><strong>Spike Tests:</strong> {len(analysis['spike_results'])} conducted</p>
                <p><strong>Peak Spike:</strong> {max((r.get('peak_load', 0) for r in analysis['spike_results'])) if analysis['spike_results'] else 0} users</p>
                <p><strong>Recovery Time:</strong> Analyzed post-spike</p>
            </div>
            
            <div class="summary-card">
                <h4>🔄 Endurance Results</h4>
                <p><strong>Duration:</strong> {metrics['endurance_duration']:.1f} hours</p>
                <p><strong>Avg Latency:</strong> {metrics['endurance_avg_latency']:.2f}s</p>
                <p><strong>Consistency:</strong> {metrics['endurance_success_rate']:.1f}% success rate</p>
            </div>
            
            <div class="summary-card">
                <h4>🛡️ Stability Analysis</h4>
                <p><strong>Error Rate:</strong> {metrics['overall_error_rate']:.2f}%</p>
                <p><strong>Memory Stability:</strong> {'⚠️ Watch' if metrics['memory_growth_rate'] > 100 else '✅ Good'}</p>
                <p><strong>Load Tolerance:</strong> Up to {metrics['breaking_point_load']} users</p>
            </div>
        </div>
        
        <div class="charts-grid">
            <div class="chart-container">
                <h3>📊 Load vs Response Time</h3>
                <canvas id="loadLatencyChart"></canvas>
            </div>
            
            <div class="chart-container">
                <h3>🚀 Throughput Analysis</h3>
                <canvas id="throughputChart"></canvas>
            </div>
            
            <div class="chart-container">
                <h3>❌ Error Rate Under Load</h3>
                <canvas id="errorRateChart"></canvas>
            </div>
            
            <div class="chart-container">
                <h3>⚡ Spike Test Results</h3>
                <canvas id="spikeChart"></canvas>
            </div>
            
            <div class="chart-container">
                <h3>🧠 Memory Usage Over Time</h3>
                <canvas id="memoryChart"></canvas>
            </div>
            
            <div class="chart-container">
                <h3>📈 Performance Degradation</h3>
                <canvas id="degradationChart"></canvas>
            </div>
        </div>
        
        <div class="analysis-section">
            <h2>🔍 Stress Test Analysis & Insights</h2>
            
            <h3>🎯 Key Findings:</h3>
            <ul style="margin: 20px 0; padding-left: 20px;">
                <li><strong>Breaking Point:</strong> System reached its limit at {metrics['breaking_point_load']} concurrent users</li>
                <li><strong>Reason for Failure:</strong> {metrics['breaking_point_reason']}</li>
                <li><strong>Stable Performance:</strong> Best performance observed at lower load levels</li>
                <li><strong>Memory Behavior:</strong> {metrics['memory_trend']} trend with {metrics['memory_growth_rate']:.1f} MB/hour growth</li>
                <li><strong>Endurance Capability:</strong> Maintained {metrics['endurance_success_rate']:.1f}% success rate during {metrics['endurance_duration']:.1f} hour test</li>
                <li><strong>Peak Efficiency:</strong> Achieved {metrics['peak_throughput']:.2f} req/s maximum throughput</li>
            </ul>
            
            <div class="recommendations">
                <h3>🎯 Performance Optimization Recommendations</h3>
                <ul>
                    {"".join(f"<li>{rec}</li>" for rec in recommendations)}
                    <li>Implement circuit breakers to prevent system overload</li>
                    <li>Set up auto-scaling triggers before breaking point</li>
                    <li>Monitor memory growth patterns in production</li>
                    <li>Consider load balancing across multiple vLLM instances</li>
                    <li>Implement graceful degradation strategies for high load</li>
                </ul>
            </div>
            
            <div style="background: #e8f5e8; border: 1px solid #4caf50; border-radius: 10px; padding: 20px; margin: 20px 0;">
                <h3 style="color: #2e7d32; margin-bottom: 15px;">✅ Production Deployment Guidelines</h3>
                <p><strong>Recommended Max Load:</strong> {int(metrics['breaking_point_load'] * 0.7) if isinstance(metrics['breaking_point_load'], (int, float)) else 'TBD'} concurrent users (70% of breaking point)</p>
                <p><strong>Monitoring Thresholds:</strong> Alert when latency > {metrics['avg_latency'] * 1.5:.1f}s or error rate > 2%</p>
                <p><strong>Scaling Strategy:</strong> Auto-scale at 60% of breaking point load</p>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')} | Scenario-4 Stress Testing Analysis</p>
            <p>🔥 This report analyzes vLLM's behavior under extreme load conditions</p>
            <p>⚠️ Ollama failed to complete this stress test scenario</p>
        </div>
    </div>

    <script>
        const chartData = {json.dumps(chart_data)};
        
        // Load vs Latency Chart
        const loadLatencyCtx = document.getElementById('loadLatencyChart').getContext('2d');
        new Chart(loadLatencyCtx, {{
            type: 'line',
            data: {{
                labels: chartData.load_levels,
                datasets: [{{
                    label: 'Average Latency (s)',
                    data: chartData.latencies,
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    tension: 0.4,
                    fill: true
                }}, {{
                    label: 'P95 Latency (s)',
                    data: chartData.p95_latencies,
                    borderColor: '#c0392b',
                    backgroundColor: 'rgba(192, 57, 43, 0.1)',
                    tension: 0.4,
                    borderDash: [5, 5]
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Latency increases with load - Shows breaking point'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Response Time (seconds)'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Concurrent Users'
                        }}
                    }}
                }}
            }}
        }});
        
        // Throughput Chart
        const throughputCtx = document.getElementById('throughputChart').getContext('2d');
        new Chart(throughputCtx, {{
            type: 'bar',
            data: {{
                labels: chartData.load_levels,
                datasets: [{{
                    label: 'Throughput (req/s)',
                    data: chartData.throughputs,
                    backgroundColor: 'rgba(39, 174, 96, 0.8)',
                    borderColor: '#27ae60',
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Throughput Under Different Load Levels'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Requests per Second'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Concurrent Users'
                        }}
                    }}
                }}
            }}
        }});
        
        // Error Rate Chart
        const errorRateCtx = document.getElementById('errorRateChart').getContext('2d');
        new Chart(errorRateCtx, {{
            type: 'line',
            data: {{
                labels: chartData.load_levels,
                datasets: [{{
                    label: 'Error Rate (%)',
                    data: chartData.error_rates,
                    borderColor: '#e67e22',
                    backgroundColor: 'rgba(230, 126, 34, 0.2)',
                    tension: 0.4,
                    fill: true
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Error Rate Progression - System Stability'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100,
                        title: {{
                            display: true,
                            text: 'Error Rate (%)'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Concurrent Users'
                        }}
                    }}
                }}
            }}
        }});
        
        // Spike Test Chart
        const spikeCtx = document.getElementById('spikeChart').getContext('2d');
        new Chart(spikeCtx, {{
            type: 'radar',
            data: {{
                labels: chartData.spike_labels,
                datasets: [{{
                    label: 'Spike Latency (s)',
                    data: chartData.spike_latencies,
                    borderColor: '#9b59b6',
                    backgroundColor: 'rgba(155, 89, 182, 0.2)',
                    pointBackgroundColor: '#9b59b6'
                }}, {{
                    label: 'Spike Throughput (req/s)',
                    data: chartData.spike_throughputs,
                    borderColor: '#f39c12',
                    backgroundColor: 'rgba(243, 156, 18, 0.2)',
                    pointBackgroundColor: '#f39c12'
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Spike Test Performance Analysis'
                    }}
                }},
                scales: {{
                    r: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
        
        // Memory Chart
        const memoryCtx = document.getElementById('memoryChart').getContext('2d');
        new Chart(memoryCtx, {{
            type: 'line',
            data: {{
                labels: chartData.memory_timestamps.slice(0, 10), // İlk 10 timestamp
                datasets: [{{
                    label: 'Memory Usage (MB)',
                    data: chartData.memory_usage.slice(0, 10),
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    tension: 0.4,
                    fill: true
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Memory Usage During Test - Growth Pattern'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Memory (MB)'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Time'
                        }}
                    }}
                }}
            }}
        }});
        
        // Performance Degradation Chart
        const degradationCtx = document.getElementById('degradationChart').getContext('2d');
        new Chart(degradationCtx, {{
            type: 'doughnut',
            data: {{
                labels: ['Successful Requests', 'Failed Requests'],
                datasets: [{{
                    data: [
                        {metrics['total_successful']},
                        {metrics['total_failed']}
                    ],
                    backgroundColor: ['#27ae60', '#e74c3c'],
                    borderWidth: 3,
                    borderColor: '#fff'
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Overall Request Success vs Failure Rate'
                    }},
                    legend: {{
                        position: 'bottom'
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
        """
        
        return html_template

def main():
    """Ana fonksiyon - HTML raporu oluşturur"""
    if len(sys.argv) != 2:
        print("Kullanım: python scenario-4-reporter.py <results_directory>")
        print("Örnek: python scenario-4-reporter.py cases/results/scenario-4")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    if not os.path.exists(results_dir):
        print(f"Hata: {results_dir} dizini bulunamadı!")
        sys.exit(1)
    
    try:
        # Reporter'ı başlat
        reporter = Scenario4Reporter(results_dir)
        reporter.load_data()
        
        # HTML raporu oluştur
        html_content = reporter.generate_html_report()
        
        # HTML dosyasını kaydet
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"scenario-4_stress_test_report_{timestamp}.html"
        
        # HTML dizini oluştur
        html_dir = Path("reporters/html-reports")
        html_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = html_dir / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML raporu başarıyla oluşturuldu: {output_path}")
        print(f"🌐 Raporu görüntülemek için: file://{output_path.absolute()}")
        
        # Performans özeti yazdır
        metrics = reporter.calculate_summary_metrics()
        
        print("\n" + "="*80)
        print("🔥 STRESS TEST SUMMARY")
        print("="*80)
        print(f"📊 Total Requests: {metrics['total_requests']:,}")
        print(f"✅ Success Rate: {((metrics['total_successful']/metrics['total_requests'])*100):.1f}%")
        print(f"⚡ Average Latency: {metrics['avg_latency']:.2f}s")
        print(f"🚀 Peak Throughput: {metrics['peak_throughput']:.2f} req/s")
        print(f"🔥 Breaking Point: {metrics['breaking_point_load']} concurrent users")
        print(f"🧠 Memory Growth: {metrics['memory_growth_rate']:.1f} MB/hour")
        print(f"⏱️ Endurance: {metrics['endurance_duration']:.1f}h with {metrics['endurance_success_rate']:.1f}% success")
        print("="*80)
        print("⚠️  Ollama failed to complete this stress test scenario")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
