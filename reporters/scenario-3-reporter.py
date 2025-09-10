#!/usr/bin/env python3
"""
Scenario-3 HTML Report Generator
Ollama ve VLLM streaming performans raporlarını karşılaştıran HTML raporu oluşturur.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

class Scenario3Reporter:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.ollama_data = None
        self.vllm_data = None
        
    def load_data(self):
        """JSON verilerini yükler"""
        # Ollama verilerini yükle
        ollama_files = list(self.results_dir.glob("ollama_streaming_data_*.json"))
        if ollama_files:
            with open(ollama_files[0], 'r') as f:
                self.ollama_data = json.load(f)
        
        # VLLM verilerini yükle
        vllm_files = list(self.results_dir.glob("vllm_streaming_data_*.json"))
        if vllm_files:
            with open(vllm_files[0], 'r') as f:
                self.vllm_data = json.load(f)
                
        if not self.ollama_data or not self.vllm_data:
            raise ValueError("Ollama veya VLLM veri dosyası bulunamadı!")
    
    def generate_chart_data(self):
        """Chart.js için streaming-specific veri hazırlar"""
        levels = []
        ollama_ttft = []
        vllm_ttft = []
        ollama_completion_time = []
        vllm_completion_time = []
        ollama_throughput = []
        vllm_throughput = []
        ollama_tokens_per_sec = []
        vllm_tokens_per_sec = []
        ollama_error_rate = []
        vllm_error_rate = []
        ollama_chunks = []
        vllm_chunks = []
        ollama_success_rate = []
        vllm_success_rate = []
        
        for result in self.ollama_data['results']:
            levels.append(result['level'])
            ollama_ttft.append(result['avg_time_to_first_token'])
            ollama_completion_time.append(result['avg_completion_time'])
            ollama_throughput.append(result['throughput_requests_per_second'])
            ollama_tokens_per_sec.append(result['avg_tokens_per_second'])
            ollama_error_rate.append(result['error_rate'])
            ollama_chunks.append(result['streaming_quality_stats']['avg_chunks'])
            ollama_success_rate.append(100 - result['error_rate'])
        
        for result in self.vllm_data['results']:
            vllm_ttft.append(result['avg_time_to_first_token'])
            vllm_completion_time.append(result['avg_completion_time'])
            vllm_throughput.append(result['throughput_requests_per_second'])
            vllm_tokens_per_sec.append(result['avg_tokens_per_second'])
            vllm_error_rate.append(result['error_rate'])
            vllm_chunks.append(result['streaming_quality_stats']['avg_chunks'])
            vllm_success_rate.append(100 - result['error_rate'])
        
        return {
            'levels': levels,
            'ollama_ttft': ollama_ttft,
            'vllm_ttft': vllm_ttft,
            'ollama_completion_time': ollama_completion_time,
            'vllm_completion_time': vllm_completion_time,
            'ollama_throughput': ollama_throughput,
            'vllm_throughput': vllm_throughput,
            'ollama_tokens_per_sec': ollama_tokens_per_sec,
            'vllm_tokens_per_sec': vllm_tokens_per_sec,
            'ollama_error_rate': ollama_error_rate,
            'vllm_error_rate': vllm_error_rate,
            'ollama_chunks': ollama_chunks,
            'vllm_chunks': vllm_chunks,
            'ollama_success_rate': ollama_success_rate,
            'vllm_success_rate': vllm_success_rate
        }
    
    def calculate_performance_metrics(self):
        """Streaming performans metriklerini hesaplar"""
        metrics = {
            'ollama': {
                'total_requests': 0, 
                'total_successful': 0, 
                'avg_ttft': 0, 
                'avg_throughput': 0,
                'avg_tokens_per_sec': 0,
                'avg_chunks': 0,
                'total_streaming_requests': 0,
                'overall_error_rate': 0
            },
            'vllm': {
                'total_requests': 0, 
                'total_successful': 0, 
                'avg_ttft': 0, 
                'avg_throughput': 0,
                'avg_tokens_per_sec': 0,
                'avg_chunks': 0,
                'total_streaming_requests': 0,
                'overall_error_rate': 0
            }
        }
        
        # Ollama metrikleri
        total_ttft_ollama = 0
        total_throughput_ollama = 0
        total_tokens_ollama = 0
        total_chunks_ollama = 0
        total_streaming_ollama = 0
        total_errors_ollama = 0
        
        for result in self.ollama_data['results']:
            metrics['ollama']['total_requests'] += result['total_requests']
            metrics['ollama']['total_successful'] += result['successful_requests']
            total_ttft_ollama += result['avg_time_to_first_token']
            total_throughput_ollama += result['throughput_requests_per_second']
            total_tokens_ollama += result['avg_tokens_per_second']
            total_chunks_ollama += result['streaming_quality_stats']['avg_chunks']
            total_streaming_ollama += result['streaming_quality_stats']['total_streaming_requests']
            total_errors_ollama += result['failed_requests']
            
        metrics['ollama']['avg_ttft'] = total_ttft_ollama / len(self.ollama_data['results'])
        metrics['ollama']['avg_throughput'] = total_throughput_ollama / len(self.ollama_data['results'])
        metrics['ollama']['avg_tokens_per_sec'] = total_tokens_ollama / len(self.ollama_data['results'])
        metrics['ollama']['avg_chunks'] = total_chunks_ollama / len(self.ollama_data['results'])
        metrics['ollama']['total_streaming_requests'] = total_streaming_ollama
        metrics['ollama']['overall_error_rate'] = (total_errors_ollama / metrics['ollama']['total_requests']) * 100
        
        # VLLM metrikleri
        total_ttft_vllm = 0
        total_throughput_vllm = 0
        total_tokens_vllm = 0
        total_chunks_vllm = 0
        total_streaming_vllm = 0
        total_errors_vllm = 0
        
        for result in self.vllm_data['results']:
            metrics['vllm']['total_requests'] += result['total_requests']
            metrics['vllm']['total_successful'] += result['successful_requests']
            total_ttft_vllm += result['avg_time_to_first_token']
            total_throughput_vllm += result['throughput_requests_per_second']
            total_tokens_vllm += result['avg_tokens_per_second']
            total_chunks_vllm += result['streaming_quality_stats']['avg_chunks']
            total_streaming_vllm += result['streaming_quality_stats']['total_streaming_requests']
            total_errors_vllm += result['failed_requests']
            
        metrics['vllm']['avg_ttft'] = total_ttft_vllm / len(self.vllm_data['results'])
        metrics['vllm']['avg_throughput'] = total_throughput_vllm / len(self.vllm_data['results'])
        metrics['vllm']['avg_tokens_per_sec'] = total_tokens_vllm / len(self.vllm_data['results'])
        metrics['vllm']['avg_chunks'] = total_chunks_vllm / len(self.vllm_data['results'])
        metrics['vllm']['total_streaming_requests'] = total_streaming_vllm
        metrics['vllm']['overall_error_rate'] = (total_errors_vllm / metrics['vllm']['total_requests']) * 100
        
        return metrics
    
    def generate_streaming_analysis(self):
        """Streaming-specific analiz ve öneriler oluşturur"""
        metrics = self.calculate_performance_metrics()
        
        analysis = {
            'winner_ttft': 'VLLM' if metrics['vllm']['avg_ttft'] < metrics['ollama']['avg_ttft'] else 'Ollama',
            'winner_throughput': 'VLLM' if metrics['vllm']['avg_throughput'] > metrics['ollama']['avg_throughput'] else 'Ollama',
            'winner_reliability': 'VLLM' if metrics['vllm']['overall_error_rate'] < metrics['ollama']['overall_error_rate'] else 'Ollama',
            'winner_tokens_per_sec': 'VLLM' if metrics['vllm']['avg_tokens_per_sec'] > metrics['ollama']['avg_tokens_per_sec'] else 'Ollama'
        }
        
        # Streaming önerileri
        recommendations = []
        
        if metrics['ollama']['avg_ttft'] > 100:  # 100 saniyeden fazla TTFT
            recommendations.append("Ollama TTFT çok yüksek - OLLAMA_NUM_PARALLEL ayarını kontrol edin")
        
        if metrics['vllm']['avg_ttft'] > 50:  # 50 saniyeden fazla TTFT
            recommendations.append("VLLM TTFT optimize edilebilir - --max-num-seqs parametresini ayarlayın")
            
        if metrics['ollama']['overall_error_rate'] > 50:
            recommendations.append("Ollama yüksek hata oranı - network konfigürasyonunu kontrol edin")
            
        if metrics['vllm']['overall_error_rate'] > 10:
            recommendations.append("VLLM timeout hataları - --max-model-len parametresini azaltmayı deneyin")
        
        return analysis, recommendations

    def generate_html_report(self):
        """HTML raporu oluşturur"""
        chart_data = self.generate_chart_data()
        metrics = self.calculate_performance_metrics()
        analysis, recommendations = self.generate_streaming_analysis()
        
        html_template = f"""
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scenario-3: Ollama vs VLLM Streaming Performance Comparison</title>
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
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
            border-left: 4px solid #3498db;
        }}
        
        .info-card h3 {{
            color: #2c3e50;
            margin-bottom: 5px;
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
        
        .metric-comparison {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
        }}
        
        .ollama {{
            color: #e74c3c;
        }}
        
        .vllm {{
            color: #27ae60;
        }}
        
        .winner {{
            background: linear-gradient(45deg, #f39c12, #f1c40f);
            color: white;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.8em;
            margin-left: 10px;
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
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
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
        
        .streaming-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .streaming-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-top: 4px solid #9b59b6;
        }}
        
        .streaming-card h4 {{
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        
        .error-analysis {{
            background: #ffebee;
            border: 1px solid #ffcdd2;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }}
        
        .error-analysis h3 {{
            color: #c62828;
            margin-bottom: 15px;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Scenario-3: Streaming Performance Comparison</h1>
            <div class="subtitle">Ollama vs VLLM - Streaming & Conversational AI Performance Analysis</div>
            
            <div class="test-info">
                <div class="info-card">
                    <h3>Model</h3>
                    <p>Qwen3-4B</p>
                </div>
                <div class="info-card">
                    <h3>Test Type</h3>
                    <p>Streaming Performance</p>
                </div>
                <div class="info-card">
                    <h3>Max Tokens</h3>
                    <p>512</p>
                </div>
                <div class="info-card">
                    <h3>Temperature</h3>
                    <p>0.7</p>
                </div>
                <div class="info-card">
                    <h3>Test Duration</h3>
                    <p>180s per level</p>
                </div>
                <div class="info-card">
                    <h3>Generated</h3>
                    <p>{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                </div>
            </div>
        </div>
        
        <div class="performance-overview">
            <div class="metric-card">
                <h3>⚡ Time to First Token (TTFT)</h3>
                <div class="metric-comparison">
                    <span class="ollama metric-value">{metrics['ollama']['avg_ttft']:.1f}s</span>
                    <span class="vllm metric-value">{metrics['vllm']['avg_ttft']:.1f}s</span>
                </div>
                <div class="winner">🏆 {analysis['winner_ttft']} Wins!</div>
                <p style="margin-top: 10px; font-size: 0.9em; color: #7f8c8d;">
                    Lower is better for streaming responsiveness
                </p>
            </div>
            
            <div class="metric-card">
                <h3>📊 Request Throughput</h3>
                <div class="metric-comparison">
                    <span class="ollama metric-value">{metrics['ollama']['avg_throughput']:.3f} req/s</span>
                    <span class="vllm metric-value">{metrics['vllm']['avg_throughput']:.3f} req/s</span>
                </div>
                <div class="winner">🏆 {analysis['winner_throughput']} Wins!</div>
                <p style="margin-top: 10px; font-size: 0.9em; color: #7f8c8d;">
                    Higher throughput = better scalability
                </p>
            </div>
            
            <div class="metric-card">
                <h3>🎯 Reliability</h3>
                <div class="metric-comparison">
                    <span class="ollama metric-value">{100 - metrics['ollama']['overall_error_rate']:.1f}%</span>
                    <span class="vllm metric-value">{100 - metrics['vllm']['overall_error_rate']:.1f}%</span>
                </div>
                <div class="winner">🏆 {analysis['winner_reliability']} Wins!</div>
                <p style="margin-top: 10px; font-size: 0.9em; color: #7f8c8d;">
                    Success rate in handling requests
                </p>
            </div>
            
            <div class="metric-card">
                <h3>🔄 Token Generation Speed</h3>
                <div class="metric-comparison">
                    <span class="ollama metric-value">{metrics['ollama']['avg_tokens_per_sec']:.2f} t/s</span>
                    <span class="vllm metric-value">{metrics['vllm']['avg_tokens_per_sec']:.2f} t/s</span>
                </div>
                <div class="winner">🏆 {analysis['winner_tokens_per_sec']} Wins!</div>
                <p style="margin-top: 10px; font-size: 0.9em; color: #7f8c8d;">
                    Token generation rate during streaming
                </p>
            </div>
        </div>

        <div class="streaming-stats">
            <div class="streaming-card">
                <h4>📈 Ollama Streaming Stats</h4>
                <p><strong>Total Requests:</strong> {metrics['ollama']['total_requests']:,}</p>
                <p><strong>Successful:</strong> {metrics['ollama']['total_successful']:,}</p>
                <p><strong>Streaming Requests:</strong> {metrics['ollama']['total_streaming_requests']:,}</p>
                <p><strong>Avg Chunks:</strong> {metrics['ollama']['avg_chunks']:.1f}</p>
                <p><strong>Error Rate:</strong> {metrics['ollama']['overall_error_rate']:.1f}%</p>
            </div>
            
            <div class="streaming-card">
                <h4>📊 VLLM Streaming Stats</h4>
                <p><strong>Total Requests:</strong> {metrics['vllm']['total_requests']:,}</p>
                <p><strong>Successful:</strong> {metrics['vllm']['total_successful']:,}</p>
                <p><strong>Streaming Requests:</strong> {metrics['vllm']['total_streaming_requests']:,}</p>
                <p><strong>Avg Chunks:</strong> {metrics['vllm']['avg_chunks']:.1f}</p>
                <p><strong>Error Rate:</strong> {metrics['vllm']['overall_error_rate']:.1f}%</p>
            </div>
        </div>
        
        <div class="charts-grid">
            <div class="chart-container">
                <h3>⚡ Time to First Token Comparison</h3>
                <canvas id="ttftChart"></canvas>
            </div>
            
            <div class="chart-container">
                <h3>⏱️ Completion Time Analysis</h3>
                <canvas id="completionTimeChart"></canvas>
            </div>
            
            <div class="chart-container">
                <h3>📊 Throughput Comparison</h3>
                <canvas id="throughputChart"></canvas>
            </div>
            
            <div class="chart-container">
                <h3>🎯 Success Rate Analysis</h3>
                <canvas id="successRateChart"></canvas>
            </div>
            
            <div class="chart-container">
                <h3>🔄 Token Generation Speed</h3>
                <canvas id="tokensChart"></canvas>
            </div>
            
            <div class="chart-container">
                <h3>📦 Streaming Chunks Analysis</h3>
                <canvas id="chunksChart"></canvas>
            </div>
        </div>
        
        <div class="analysis-section">
            <h2>🔍 Streaming Performance Analysis</h2>
            
            <div class="error-analysis">
                <h3>❌ Error Analysis</h3>
                <p><strong>Ollama Primary Error:</strong> Network connection issues (nodename nor servname provided)</p>
                <p><strong>VLLM Primary Error:</strong> HTTP timeout errors during high load</p>
                <p><strong>Key Insight:</strong> VLLM shows much better reliability with {100 - metrics['vllm']['overall_error_rate']:.1f}% success rate vs Ollama's {100 - metrics['ollama']['overall_error_rate']:.1f}%</p>
            </div>
            
            <h3>📊 Key Findings:</h3>
            <ul style="margin: 20px 0; padding-left: 20px;">
                <li><strong>TTFT Performance:</strong> VLLM significantly outperforms Ollama with {metrics['vllm']['avg_ttft']:.1f}s vs {metrics['ollama']['avg_ttft']:.1f}s average time to first token</li>
                <li><strong>Reliability:</strong> VLLM shows superior stability with {100 - metrics['vllm']['overall_error_rate']:.1f}% success rate</li>
                <li><strong>Throughput:</strong> Both systems show low throughput, but VLLM maintains better consistency</li>
                <li><strong>Streaming Quality:</strong> Ollama generates more chunks per request ({metrics['ollama']['avg_chunks']:.1f} vs {metrics['vllm']['avg_chunks']:.1f})</li>
            </ul>
            
            <div class="recommendations">
                <h3>🎯 Optimization Recommendations</h3>
                <ul>
                    {"".join(f"<li>{rec}</li>" for rec in recommendations)}
                    <li>VLLM için --enable-chunked-prefill kullanarak streaming performansını artırın</li>
                    <li>Ollama için OLLAMA_FLASH_ATTENTION=1 ile token generation hızını artırın</li>
                    <li>Production ortamında WebSocket connections kullanmayı değerlendirin</li>
                    <li>Client-side chunking ile kullanıcı deneyimini iyileştirin</li>
                </ul>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')} | Scenario-3 Streaming Performance Analysis</p>
            <p>🚀 This report analyzes streaming capabilities of Ollama vs VLLM for real-time AI applications</p>
        </div>
    </div>

    <script>
        const chartData = {json.dumps(chart_data)};
        
        // TTFT Chart
        const ttftCtx = document.getElementById('ttftChart').getContext('2d');
        new Chart(ttftCtx, {{
            type: 'line',
            data: {{
                labels: chartData.levels,
                datasets: [{{
                    label: 'Ollama TTFT (s)',
                    data: chartData.ollama_ttft,
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    tension: 0.4,
                    fill: true
                }}, {{
                    label: 'VLLM TTFT (s)',
                    data: chartData.vllm_ttft,
                    borderColor: '#27ae60',
                    backgroundColor: 'rgba(39, 174, 96, 0.1)',
                    tension: 0.4,
                    fill: true
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Time to First Token - Lower is Better'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Time (seconds)'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Concurrency Level'
                        }}
                    }}
                }}
            }}
        }});
        
        // Completion Time Chart
        const completionCtx = document.getElementById('completionTimeChart').getContext('2d');
        new Chart(completionCtx, {{
            type: 'bar',
            data: {{
                labels: chartData.levels,
                datasets: [{{
                    label: 'Ollama Completion Time (s)',
                    data: chartData.ollama_completion_time,
                    backgroundColor: 'rgba(231, 76, 60, 0.8)',
                    borderColor: '#e74c3c',
                    borderWidth: 2
                }}, {{
                    label: 'VLLM Completion Time (s)',
                    data: chartData.vllm_completion_time,
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
                        text: 'Average Completion Time Comparison'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Time (seconds)'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Concurrency Level'
                        }}
                    }}
                }}
            }}
        }});
        
        // Throughput Chart
        const throughputCtx = document.getElementById('throughputChart').getContext('2d');
        new Chart(throughputCtx, {{
            type: 'line',
            data: {{
                labels: chartData.levels,
                datasets: [{{
                    label: 'Ollama Throughput (req/s)',
                    data: chartData.ollama_throughput,
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    tension: 0.4,
                    fill: true
                }}, {{
                    label: 'VLLM Throughput (req/s)',
                    data: chartData.vllm_throughput,
                    borderColor: '#27ae60',
                    backgroundColor: 'rgba(39, 174, 96, 0.1)',
                    tension: 0.4,
                    fill: true
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Request Throughput - Higher is Better'
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
                            text: 'Concurrency Level'
                        }}
                    }}
                }}
            }}
        }});
        
        // Success Rate Chart
        const successCtx = document.getElementById('successRateChart').getContext('2d');
        new Chart(successCtx, {{
            type: 'doughnut',
            data: {{
                labels: ['Ollama Success Rate', 'VLLM Success Rate'],
                datasets: [{{
                    data: [
                        {100 - metrics['ollama']['overall_error_rate']:.1f},
                        {100 - metrics['vllm']['overall_error_rate']:.1f}
                    ],
                    backgroundColor: ['#e74c3c', '#27ae60'],
                    borderWidth: 3,
                    borderColor: '#fff'
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Overall Success Rate Comparison (%)'
                    }},
                    legend: {{
                        position: 'bottom'
                    }}
                }}
            }}
        }});
        
        // Tokens per Second Chart
        const tokensCtx = document.getElementById('tokensChart').getContext('2d');
        new Chart(tokensCtx, {{
            type: 'radar',
            data: {{
                labels: chartData.levels.map(l => `Level ${{l}}`),
                datasets: [{{
                    label: 'Ollama Tokens/s',
                    data: chartData.ollama_tokens_per_sec,
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.2)',
                    pointBackgroundColor: '#e74c3c'
                }}, {{
                    label: 'VLLM Tokens/s',
                    data: chartData.vllm_tokens_per_sec,
                    borderColor: '#27ae60',
                    backgroundColor: 'rgba(39, 174, 96, 0.2)',
                    pointBackgroundColor: '#27ae60'
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Token Generation Speed Radar'
                    }}
                }},
                scales: {{
                    r: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Tokens per Second'
                        }}
                    }}
                }}
            }}
        }});
        
        // Chunks Chart
        const chunksCtx = document.getElementById('chunksChart').getContext('2d');
        new Chart(chunksCtx, {{
            type: 'bar',
            data: {{
                labels: chartData.levels,
                datasets: [{{
                    label: 'Ollama Avg Chunks',
                    data: chartData.ollama_chunks,
                    backgroundColor: 'rgba(231, 76, 60, 0.8)',
                    borderColor: '#e74c3c',
                    borderWidth: 2
                }}, {{
                    label: 'VLLM Avg Chunks',
                    data: chartData.vllm_chunks,
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
                        text: 'Average Streaming Chunks per Request'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Average Chunks'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Concurrency Level'
                        }}
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
        print("Kullanım: python scenario-3-reporter.py <results_directory>")
        print("Örnek: python scenario-3-reporter.py cases/results/scenario-3")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    if not os.path.exists(results_dir):
        print(f"Hata: {results_dir} dizini bulunamadı!")
        sys.exit(1)
    
    try:
        # Reporter'ı başlat
        reporter = Scenario3Reporter(results_dir)
        reporter.load_data()
        
        # HTML raporu oluştur
        html_content = reporter.generate_html_report()
        
        # HTML dosyasını kaydet
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"scenario-3_comparison_report_{timestamp}.html"
        
        # HTML dizini oluştur
        html_dir = Path("reporters/html-reports")
        html_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = html_dir / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML raporu başarıyla oluşturuldu: {output_path}")
        print(f"🌐 Raporu görüntülemek için: file://{output_path.absolute()}")
        
        # Performans özeti yazdır
        reporter_instance = reporter
        metrics = reporter_instance.calculate_performance_metrics()
        analysis, _ = reporter_instance.generate_streaming_analysis()
        
        print("\n" + "="*80)
        print("📊 STREAMING PERFORMANCE SUMMARY")
        print("="*80)
        print(f"⚡ Time to First Token Winner: {analysis['winner_ttft']}")
        print(f"📈 Throughput Winner: {analysis['winner_throughput']}")
        print(f"🎯 Reliability Winner: {analysis['winner_reliability']}")
        print(f"🔄 Token Speed Winner: {analysis['winner_tokens_per_sec']}")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
