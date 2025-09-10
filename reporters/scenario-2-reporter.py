#!/usr/bin/env python3
"""
Scenario-2 HTML Report Generator
Ollama ve VLLM complex reasoning performans raporlarını karşılaştıran HTML raporu oluşturur.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

class Scenario2Reporter:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.ollama_data = None
        self.vllm_data = None
        
    def load_data(self):
        """JSON verilerini yükler"""
        # Ollama verilerini yükle
        ollama_files = list(self.results_dir.glob("ollama_complex_reasoning_data_*.json"))
        if ollama_files:
            with open(ollama_files[0], 'r') as f:
                self.ollama_data = json.load(f)
        
        # VLLM verilerini yükle
        vllm_files = list(self.results_dir.glob("vllm_complex_reasoning_data_*.json"))
        if vllm_files:
            with open(vllm_files[0], 'r') as f:
                self.vllm_data = json.load(f)
                
        if not self.ollama_data or not self.vllm_data:
            raise ValueError("Ollama veya VLLM veri dosyası bulunamadı!")
    
    def generate_chart_data(self):
        """Chart.js için veri hazırlar"""
        levels = []
        ollama_completion_time = []
        vllm_completion_time = []
        ollama_throughput = []
        vllm_throughput = []
        ollama_accuracy = []
        vllm_accuracy = []
        ollama_math_accuracy = []
        vllm_math_accuracy = []
        ollama_code_accuracy = []
        vllm_code_accuracy = []
        ollama_qa_accuracy = []
        vllm_qa_accuracy = []
        ollama_error_rate = []
        vllm_error_rate = []
        ollama_tokens_per_sec = []
        vllm_tokens_per_sec = []
        
        for result in self.ollama_data['results']:
            levels.append(result['level'])
            ollama_completion_time.append(result['avg_completion_time'])
            ollama_throughput.append(result['throughput'])
            ollama_accuracy.append(result['overall_accuracy'] * 100)
            ollama_math_accuracy.append(result['math_accuracy'] * 100)
            ollama_code_accuracy.append(result['code_accuracy'] * 100)
            ollama_qa_accuracy.append(result['qa_accuracy'] * 100)
            ollama_error_rate.append(result['error_rate'])
            ollama_tokens_per_sec.append(result['avg_tokens_per_second'])
        
        for result in self.vllm_data['results']:
            vllm_completion_time.append(result['avg_completion_time'])
            vllm_throughput.append(result['throughput'])
            vllm_accuracy.append(result['overall_accuracy'] * 100)
            vllm_math_accuracy.append(result['math_accuracy'] * 100)
            vllm_code_accuracy.append(result['code_accuracy'] * 100)
            vllm_qa_accuracy.append(result['qa_accuracy'] * 100)
            vllm_error_rate.append(result['error_rate'])
            vllm_tokens_per_sec.append(result['avg_tokens_per_second'])
        
        return {
            'levels': levels,
            'ollama_completion_time': ollama_completion_time,
            'vllm_completion_time': vllm_completion_time,
            'ollama_throughput': ollama_throughput,
            'vllm_throughput': vllm_throughput,
            'ollama_accuracy': ollama_accuracy,
            'vllm_accuracy': vllm_accuracy,
            'ollama_math_accuracy': ollama_math_accuracy,
            'vllm_math_accuracy': vllm_math_accuracy,
            'ollama_code_accuracy': ollama_code_accuracy,
            'vllm_code_accuracy': vllm_code_accuracy,
            'ollama_qa_accuracy': ollama_qa_accuracy,
            'vllm_qa_accuracy': vllm_qa_accuracy,
            'ollama_error_rate': ollama_error_rate,
            'vllm_error_rate': vllm_error_rate,
            'ollama_tokens_per_sec': ollama_tokens_per_sec,
            'vllm_tokens_per_sec': vllm_tokens_per_sec
        }
    
    def calculate_performance_metrics(self):
        """Performans metriklerini hesaplar"""
        metrics = {
            'ollama': {
                'total_tasks': 0, 
                'total_successful': 0, 
                'avg_throughput': 0, 
                'avg_tokens_per_sec': 0,
                'avg_accuracy': 0,
                'avg_math_accuracy': 0,
                'avg_code_accuracy': 0,
                'avg_qa_accuracy': 0,
                'avg_error_rate': 0,
                'avg_reasoning_steps': 0,
                'avg_completion_time': 0
            },
            'vllm': {
                'total_tasks': 0, 
                'total_successful': 0, 
                'avg_throughput': 0, 
                'avg_tokens_per_sec': 0,
                'avg_accuracy': 0,
                'avg_math_accuracy': 0,
                'avg_code_accuracy': 0,
                'avg_qa_accuracy': 0,
                'avg_error_rate': 0,
                'avg_reasoning_steps': 0,
                'avg_completion_time': 0
            }
        }
        
        # Ollama metrikleri
        for result in self.ollama_data['results']:
            metrics['ollama']['total_tasks'] += result['total_tasks']
            metrics['ollama']['total_successful'] += result['successful_tasks']
            
        metrics['ollama']['avg_throughput'] = sum(r['throughput'] for r in self.ollama_data['results']) / len(self.ollama_data['results'])
        metrics['ollama']['avg_tokens_per_sec'] = sum(r['avg_tokens_per_second'] for r in self.ollama_data['results']) / len(self.ollama_data['results'])
        metrics['ollama']['avg_accuracy'] = sum(r['overall_accuracy'] for r in self.ollama_data['results']) / len(self.ollama_data['results']) * 100
        metrics['ollama']['avg_math_accuracy'] = sum(r['math_accuracy'] for r in self.ollama_data['results']) / len(self.ollama_data['results']) * 100
        metrics['ollama']['avg_code_accuracy'] = sum(r['code_accuracy'] for r in self.ollama_data['results']) / len(self.ollama_data['results']) * 100
        metrics['ollama']['avg_qa_accuracy'] = sum(r['qa_accuracy'] for r in self.ollama_data['results']) / len(self.ollama_data['results']) * 100
        metrics['ollama']['avg_error_rate'] = sum(r['error_rate'] for r in self.ollama_data['results']) / len(self.ollama_data['results'])
        metrics['ollama']['avg_reasoning_steps'] = sum(r['avg_reasoning_steps'] for r in self.ollama_data['results']) / len(self.ollama_data['results'])
        metrics['ollama']['avg_completion_time'] = sum(r['avg_completion_time'] for r in self.ollama_data['results']) / len(self.ollama_data['results'])
        
        # VLLM metrikleri
        for result in self.vllm_data['results']:
            metrics['vllm']['total_tasks'] += result['total_tasks']
            metrics['vllm']['total_successful'] += result['successful_tasks']
            
        metrics['vllm']['avg_throughput'] = sum(r['throughput'] for r in self.vllm_data['results']) / len(self.vllm_data['results'])
        metrics['vllm']['avg_tokens_per_sec'] = sum(r['avg_tokens_per_second'] for r in self.vllm_data['results']) / len(self.vllm_data['results'])
        metrics['vllm']['avg_accuracy'] = sum(r['overall_accuracy'] for r in self.vllm_data['results']) / len(self.vllm_data['results']) * 100
        metrics['vllm']['avg_math_accuracy'] = sum(r['math_accuracy'] for r in self.vllm_data['results']) / len(self.vllm_data['results']) * 100
        metrics['vllm']['avg_code_accuracy'] = sum(r['code_accuracy'] for r in self.vllm_data['results']) / len(self.vllm_data['results']) * 100
        metrics['vllm']['avg_qa_accuracy'] = sum(r['qa_accuracy'] for r in self.vllm_data['results']) / len(self.vllm_data['results']) * 100
        metrics['vllm']['avg_error_rate'] = sum(r['error_rate'] for r in self.vllm_data['results']) / len(self.vllm_data['results'])
        metrics['vllm']['avg_reasoning_steps'] = sum(r['avg_reasoning_steps'] for r in self.vllm_data['results']) / len(self.vllm_data['results'])
        metrics['vllm']['avg_completion_time'] = sum(r['avg_completion_time'] for r in self.vllm_data['results']) / len(self.vllm_data['results'])
        
        return metrics

    def generate_html_report(self):
        """HTML raporu oluşturur"""
        chart_data = self.generate_chart_data()
        metrics = self.calculate_performance_metrics()
        
        html_template = f"""
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scenario-2 Complex Reasoning Comparison: Ollama vs VLLM</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .summary {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            padding: 30px;
            background: #f8f9fa;
        }}
        .summary-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 5px solid;
        }}
        .ollama-card {{
            border-left-color: #ff6b6b;
        }}
        .vllm-card {{
            border-left-color: #4ecdc4;
        }}
        .summary-card h3 {{
            margin: 0 0 15px 0;
            font-size: 1.3em;
            color: #333;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }}
        .metric:last-child {{
            border-bottom: none;
        }}
        .metric-label {{
            color: #666;
        }}
        .metric-value {{
            font-weight: bold;
            color: #333;
        }}
        .charts {{
            padding: 30px;
        }}
        .chart-container {{
            margin: 30px 0;
            height: 400px;
            position: relative;
        }}
        .chart-title {{
            text-align: center;
            font-size: 1.4em;
            margin-bottom: 20px;
            color: #333;
            font-weight: 500;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin: 30px 0;
        }}
        .chart-container-small {{
            height: 300px;
            position: relative;
        }}
        .detailed-table {{
            padding: 30px;
            background: #f8f9fa;
        }}
        .table-title {{
            font-size: 1.5em;
            margin-bottom: 20px;
            color: #333;
            text-align: center;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 500;
            font-size: 0.9em;
        }}
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
            font-size: 0.9em;
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .performance-indicator {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .better {{
            background: #d4edda;
            color: #155724;
        }}
        .worse {{
            background: #f8d7da;
            color: #721c24;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            background: #f8f9fa;
            border-top: 1px solid #eee;
        }}
        .comparison-highlights {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: white;
        }}
        .highlight-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .highlight-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .highlight-label {{
            opacity: 0.9;
            font-size: 0.9em;
        }}
        .error-indicator {{
            background: #f8d7da;
            color: #721c24;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.8em;
        }}
        .success-indicator {{
            background: #d4edda;
            color: #155724;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.8em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Scenario-2 Complex Reasoning Report</h1>
            <p>Ollama vs VLLM Complex Reasoning Performance Comparison</p>
            <p>Model: {self.ollama_data['config']['model_name']} vs {self.vllm_data['config']['model_name']} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="summary">
            <div class="summary-card ollama-card">
                <h3>🦙 Ollama Performance</h3>
                <div class="metric">
                    <span class="metric-label">Total Tasks:</span>
                    <span class="metric-value">{metrics['ollama']['total_tasks']:,}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Successful Tasks:</span>
                    <span class="metric-value">{metrics['ollama']['total_successful']:,}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Average Accuracy:</span>
                    <span class="metric-value">{metrics['ollama']['avg_accuracy']:.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Average Throughput:</span>
                    <span class="metric-value">{metrics['ollama']['avg_throughput']:.3f} tasks/s</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Average Tokens/sec:</span>
                    <span class="metric-value">{metrics['ollama']['avg_tokens_per_sec']:.1f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Average Error Rate:</span>
                    <span class="metric-value">{metrics['ollama']['avg_error_rate']:.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Reasoning Steps:</span>
                    <span class="metric-value">{metrics['ollama']['avg_reasoning_steps']:.1f}</span>
                </div>
            </div>

            <div class="summary-card vllm-card">
                <h3>⚡ VLLM Performance</h3>
                <div class="metric">
                    <span class="metric-label">Total Tasks:</span>
                    <span class="metric-value">{metrics['vllm']['total_tasks']:,}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Successful Tasks:</span>
                    <span class="metric-value">{metrics['vllm']['total_successful']:,}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Average Accuracy:</span>
                    <span class="metric-value">{metrics['vllm']['avg_accuracy']:.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Average Throughput:</span>
                    <span class="metric-value">{metrics['vllm']['avg_throughput']:.3f} tasks/s</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Average Tokens/sec:</span>
                    <span class="metric-value">{metrics['vllm']['avg_tokens_per_sec']:.1f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Average Error Rate:</span>
                    <span class="metric-value">{metrics['vllm']['avg_error_rate']:.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Reasoning Steps:</span>
                    <span class="metric-value">{metrics['vllm']['avg_reasoning_steps']:.1f}</span>
                </div>
            </div>
        </div>

        <div class="comparison-highlights">
            <div class="highlight-card">
                <div class="highlight-label">Accuracy Winner</div>
                <div class="highlight-value">{"VLLM" if metrics['vllm']['avg_accuracy'] > metrics['ollama']['avg_accuracy'] else "Ollama"}</div>
                <div class="highlight-label">{max(metrics['vllm']['avg_accuracy'], metrics['ollama']['avg_accuracy']):.1f}%</div>
            </div>
            <div class="highlight-card">
                <div class="highlight-label">Throughput Winner</div>
                <div class="highlight-value">{"VLLM" if metrics['vllm']['avg_throughput'] > metrics['ollama']['avg_throughput'] else "Ollama"}</div>
                <div class="highlight-label">{max(metrics['vllm']['avg_throughput'], metrics['ollama']['avg_throughput']):.3f} tasks/s</div>
            </div>
            <div class="highlight-card">
                <div class="highlight-label">Reliability Winner</div>
                <div class="highlight-value">{"VLLM" if metrics['vllm']['avg_error_rate'] < metrics['ollama']['avg_error_rate'] else "Ollama"}</div>
                <div class="highlight-label">{min(metrics['vllm']['avg_error_rate'], metrics['ollama']['avg_error_rate']):.1f}% error rate</div>
            </div>
            <div class="highlight-card">
                <div class="highlight-label">Total Tasks</div>
                <div class="highlight-value">{metrics['vllm']['total_tasks'] + metrics['ollama']['total_tasks']:,}</div>
                <div class="highlight-label">Complex Reasoning Tasks</div>
            </div>
            <div class="highlight-card">
                <div class="highlight-label">Success Rate</div>
                <div class="highlight-value">{"VLLM" if (metrics['vllm']['total_successful']/metrics['vllm']['total_tasks']) > (metrics['ollama']['total_successful']/metrics['ollama']['total_tasks']) else "Ollama"}</div>
                <div class="highlight-label">{max(metrics['vllm']['total_successful']/metrics['vllm']['total_tasks'], metrics['ollama']['total_successful']/metrics['ollama']['total_tasks'])*100:.1f}%</div>
            </div>
        </div>

        <div class="charts">
            <div class="chart-title">🎯 Overall Accuracy Comparison (Higher is Better)</div>
            <div class="chart-container">
                <canvas id="accuracyChart"></canvas>
            </div>

            <div class="charts-grid">
                <div>
                    <div class="chart-title">⏱️ Completion Time (Lower is Better)</div>
                    <div class="chart-container-small">
                        <canvas id="completionTimeChart"></canvas>
                    </div>
                </div>
                <div>
                    <div class="chart-title">🚀 Throughput (Higher is Better)</div>
                    <div class="chart-container-small">
                        <canvas id="throughputChart"></canvas>
                    </div>
                </div>
            </div>

            <div class="chart-title">📊 Task Type Accuracy Comparison</div>
            <div class="charts-grid">
                <div>
                    <div class="chart-title">🧮 Math Accuracy</div>
                    <div class="chart-container-small">
                        <canvas id="mathAccuracyChart"></canvas>
                    </div>
                </div>
                <div>
                    <div class="chart-title">💻 Code Accuracy</div>
                    <div class="chart-container-small">
                        <canvas id="codeAccuracyChart"></canvas>
                    </div>
                </div>
            </div>

            <div class="charts-grid">
                <div>
                    <div class="chart-title">❓ QA Accuracy</div>
                    <div class="chart-container-small">
                        <canvas id="qaAccuracyChart"></canvas>
                    </div>
                </div>
                <div>
                    <div class="chart-title">❌ Error Rate (Lower is Better)</div>
                    <div class="chart-container-small">
                        <canvas id="errorRateChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <div class="detailed-table">
            <div class="table-title">📋 Detailed Complex Reasoning Performance Comparison</div>
            <table>
                <thead>
                    <tr>
                        <th>Level</th>
                        <th>Platform</th>
                        <th>Tasks</th>
                        <th>Success</th>
                        <th>Overall Acc.%</th>
                        <th>Math%</th>
                        <th>Code%</th>
                        <th>QA%</th>
                        <th>Avg Time(s)</th>
                        <th>TPS</th>
                        <th>Error Rate%</th>
                        <th>Tokens/s</th>
                        <th>Reasoning Steps</th>
                    </tr>
                </thead>
                <tbody>
"""

        # Detaylı tablo verilerini ekle
        for i, level in enumerate(chart_data['levels']):
            ollama_result = self.ollama_data['results'][i]
            vllm_result = self.vllm_data['results'][i]
            
            # Karşılaştırma için işaretleyiciler
            better_accuracy_ollama = ollama_result['overall_accuracy'] > vllm_result['overall_accuracy']
            better_throughput_ollama = ollama_result['throughput'] > vllm_result['throughput']
            better_error_rate_ollama = ollama_result['error_rate'] < vllm_result['error_rate']
            better_completion_time_ollama = ollama_result['avg_completion_time'] < vllm_result['avg_completion_time']
            better_tokens_ollama = ollama_result['avg_tokens_per_second'] > vllm_result['avg_tokens_per_second']
            
            html_template += f"""
                    <tr>
                        <td rowspan="2" style="vertical-align: middle; font-weight: bold; background: #f8f9fa;">{level}</td>
                        <td><strong>🦙 Ollama</strong></td>
                        <td>{ollama_result['total_tasks']}</td>
                        <td><span class="{'success-indicator' if ollama_result['successful_tasks'] > 0 else 'error-indicator'}">{ollama_result['successful_tasks']}</span></td>
                        <td><span class="performance-indicator {'better' if better_accuracy_ollama else 'worse'}">{ollama_result['overall_accuracy']*100:.1f}</span></td>
                        <td>{ollama_result['math_accuracy']*100:.1f}</td>
                        <td>{ollama_result['code_accuracy']*100:.1f}</td>
                        <td>{ollama_result['qa_accuracy']*100:.1f}</td>
                        <td><span class="performance-indicator {'better' if better_completion_time_ollama else 'worse'}">{ollama_result['avg_completion_time']:.1f}</span></td>
                        <td><span class="performance-indicator {'better' if better_throughput_ollama else 'worse'}">{ollama_result['throughput']:.3f}</span></td>
                        <td><span class="performance-indicator {'better' if better_error_rate_ollama else 'worse'}">{ollama_result['error_rate']:.1f}</span></td>
                        <td><span class="performance-indicator {'better' if better_tokens_ollama else 'worse'}">{ollama_result['avg_tokens_per_second']:.1f}</span></td>
                        <td>{ollama_result['avg_reasoning_steps']:.1f}</td>
                    </tr>
                    <tr>
                        <td><strong>⚡ VLLM</strong></td>
                        <td>{vllm_result['total_tasks']}</td>
                        <td><span class="{'success-indicator' if vllm_result['successful_tasks'] > 0 else 'error-indicator'}">{vllm_result['successful_tasks']}</span></td>
                        <td><span class="performance-indicator {'better' if not better_accuracy_ollama else 'worse'}">{vllm_result['overall_accuracy']*100:.1f}</span></td>
                        <td>{vllm_result['math_accuracy']*100:.1f}</td>
                        <td>{vllm_result['code_accuracy']*100:.1f}</td>
                        <td>{vllm_result['qa_accuracy']*100:.1f}</td>
                        <td><span class="performance-indicator {'better' if not better_completion_time_ollama else 'worse'}">{vllm_result['avg_completion_time']:.1f}</span></td>
                        <td><span class="performance-indicator {'better' if not better_throughput_ollama else 'worse'}">{vllm_result['throughput']:.3f}</span></td>
                        <td><span class="performance-indicator {'better' if not better_error_rate_ollama else 'worse'}">{vllm_result['error_rate']:.1f}</span></td>
                        <td><span class="performance-indicator {'better' if not better_tokens_ollama else 'worse'}">{vllm_result['avg_tokens_per_second']:.1f}</span></td>
                        <td>{vllm_result['avg_reasoning_steps']:.1f}</td>
                    </tr>
"""

        html_template += f"""
                </tbody>
            </table>
        </div>

        <div class="footer">
            <p>Complex reasoning report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')} | 
            Ollama Test: {self.ollama_data['timestamp'][:19]} | 
            VLLM Test: {self.vllm_data['timestamp'][:19]}</p>
        </div>
    </div>

    <script>
        // Chart konfigürasyonu
        const chartOptions = {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                legend: {{
                    position: 'top',
                }},
            }},
            scales: {{
                y: {{
                    beginAtZero: true
                }}
            }}
        }};

        // Overall Accuracy Chart
        new Chart(document.getElementById('accuracyChart'), {{
            type: 'line',
            data: {{
                labels: {chart_data['levels']},
                datasets: [{{
                    label: '🦙 Ollama',
                    data: {chart_data['ollama_accuracy']},
                    borderColor: '#ff6b6b',
                    backgroundColor: 'rgba(255, 107, 107, 0.1)',
                    tension: 0.4
                }}, {{
                    label: '⚡ VLLM',
                    data: {chart_data['vllm_accuracy']},
                    borderColor: '#4ecdc4',
                    backgroundColor: 'rgba(78, 205, 196, 0.1)',
                    tension: 0.4
                }}]
            }},
            options: {{
                ...chartOptions,
                scales: {{
                    ...chartOptions.scales,
                    y: {{
                        ...chartOptions.scales.y,
                        title: {{
                            display: true,
                            text: 'Accuracy (%)'
                        }},
                        max: 100
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

        // Completion Time Chart
        new Chart(document.getElementById('completionTimeChart'), {{
            type: 'bar',
            data: {{
                labels: {chart_data['levels']},
                datasets: [{{
                    label: '🦙 Ollama',
                    data: {chart_data['ollama_completion_time']},
                    backgroundColor: 'rgba(255, 107, 107, 0.7)',
                    borderColor: '#ff6b6b',
                    borderWidth: 2
                }}, {{
                    label: '⚡ VLLM',
                    data: {chart_data['vllm_completion_time']},
                    backgroundColor: 'rgba(78, 205, 196, 0.7)',
                    borderColor: '#4ecdc4',
                    borderWidth: 2
                }}]
            }},
            options: {{
                ...chartOptions,
                scales: {{
                    ...chartOptions.scales,
                    y: {{
                        ...chartOptions.scales.y,
                        title: {{
                            display: true,
                            text: 'Completion Time (s)'
                        }}
                    }}
                }}
            }}
        }});

        // Throughput Chart
        new Chart(document.getElementById('throughputChart'), {{
            type: 'bar',
            data: {{
                labels: {chart_data['levels']},
                datasets: [{{
                    label: '🦙 Ollama',
                    data: {chart_data['ollama_throughput']},
                    backgroundColor: 'rgba(255, 107, 107, 0.7)',
                    borderColor: '#ff6b6b',
                    borderWidth: 2
                }}, {{
                    label: '⚡ VLLM',
                    data: {chart_data['vllm_throughput']},
                    backgroundColor: 'rgba(78, 205, 196, 0.7)',
                    borderColor: '#4ecdc4',
                    borderWidth: 2
                }}]
            }},
            options: {{
                ...chartOptions,
                scales: {{
                    ...chartOptions.scales,
                    y: {{
                        ...chartOptions.scales.y,
                        title: {{
                            display: true,
                            text: 'Tasks per Second'
                        }}
                    }}
                }}
            }}
        }});

        // Math Accuracy Chart
        new Chart(document.getElementById('mathAccuracyChart'), {{
            type: 'line',
            data: {{
                labels: {chart_data['levels']},
                datasets: [{{
                    label: '🦙 Ollama',
                    data: {chart_data['ollama_math_accuracy']},
                    borderColor: '#ff6b6b',
                    backgroundColor: 'rgba(255, 107, 107, 0.1)',
                    tension: 0.4
                }}, {{
                    label: '⚡ VLLM',
                    data: {chart_data['vllm_math_accuracy']},
                    borderColor: '#4ecdc4',
                    backgroundColor: 'rgba(78, 205, 196, 0.1)',
                    tension: 0.4
                }}]
            }},
            options: {{
                ...chartOptions,
                scales: {{
                    ...chartOptions.scales,
                    y: {{
                        ...chartOptions.scales.y,
                        title: {{
                            display: true,
                            text: 'Math Accuracy (%)'
                        }},
                        max: 100
                    }}
                }}
            }}
        }});

        // Code Accuracy Chart
        new Chart(document.getElementById('codeAccuracyChart'), {{
            type: 'line',
            data: {{
                labels: {chart_data['levels']},
                datasets: [{{
                    label: '🦙 Ollama',
                    data: {chart_data['ollama_code_accuracy']},
                    borderColor: '#ff6b6b',
                    backgroundColor: 'rgba(255, 107, 107, 0.1)',
                    tension: 0.4
                }}, {{
                    label: '⚡ VLLM',
                    data: {chart_data['vllm_code_accuracy']},
                    borderColor: '#4ecdc4',
                    backgroundColor: 'rgba(78, 205, 196, 0.1)',
                    tension: 0.4
                }}]
            }},
            options: {{
                ...chartOptions,
                scales: {{
                    ...chartOptions.scales,
                    y: {{
                        ...chartOptions.scales.y,
                        title: {{
                            display: true,
                            text: 'Code Accuracy (%)'
                        }},
                        max: 100
                    }}
                }}
            }}
        }});

        // QA Accuracy Chart
        new Chart(document.getElementById('qaAccuracyChart'), {{
            type: 'line',
            data: {{
                labels: {chart_data['levels']},
                datasets: [{{
                    label: '🦙 Ollama',
                    data: {chart_data['ollama_qa_accuracy']},
                    borderColor: '#ff6b6b',
                    backgroundColor: 'rgba(255, 107, 107, 0.1)',
                    tension: 0.4
                }}, {{
                    label: '⚡ VLLM',
                    data: {chart_data['vllm_qa_accuracy']},
                    borderColor: '#4ecdc4',
                    backgroundColor: 'rgba(78, 205, 196, 0.1)',
                    tension: 0.4
                }}]
            }},
            options: {{
                ...chartOptions,
                scales: {{
                    ...chartOptions.scales,
                    y: {{
                        ...chartOptions.scales.y,
                        title: {{
                            display: true,
                            text: 'QA Accuracy (%)'
                        }},
                        max: 100
                    }}
                }}
            }}
        }});

        // Error Rate Chart
        new Chart(document.getElementById('errorRateChart'), {{
            type: 'bar',
            data: {{
                labels: {chart_data['levels']},
                datasets: [{{
                    label: '🦙 Ollama',
                    data: {chart_data['ollama_error_rate']},
                    backgroundColor: 'rgba(255, 107, 107, 0.7)',
                    borderColor: '#ff6b6b',
                    borderWidth: 2
                }}, {{
                    label: '⚡ VLLM',
                    data: {chart_data['vllm_error_rate']},
                    backgroundColor: 'rgba(78, 205, 196, 0.7)',
                    borderColor: '#4ecdc4',
                    borderWidth: 2
                }}]
            }},
            options: {{
                ...chartOptions,
                scales: {{
                    ...chartOptions.scales,
                    y: {{
                        ...chartOptions.scales.y,
                        title: {{
                            display: true,
                            text: 'Error Rate (%)'
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

    def save_report(self, output_file=None):
        """HTML raporunu dosyaya kaydeder"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"html-reports/scenario-2_comparison_report_{timestamp}.html"
        
        # html-reports klasörünü oluştur
        os.makedirs("html-reports", exist_ok=True)
        
        html_content = self.generate_html_report()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_file

def main():
    """Ana fonksiyon"""
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Varsayılan olarak scenario-2 klasörünü kullan
        script_dir = Path(__file__).parent
        results_dir = script_dir.parent / "cases" / "results" / "scenario-2"
    
    if not Path(results_dir).exists():
        print(f"❌ Hata: {results_dir} klasörü bulunamadı!")
        sys.exit(1)
    
    try:
        print("🧠 Scenario-2 Complex Reasoning HTML raporu oluşturuluyor...")
        reporter = Scenario2Reporter(results_dir)
        reporter.load_data()
        
        output_file = reporter.save_report()
        print(f"✅ HTML raporu başarıyla oluşturuldu: {output_file}")
        print(f"🌐 Raporu görüntülemek için dosyayı web tarayıcınızda açın.")
        
        # Rapor özeti
        metrics = reporter.calculate_performance_metrics()
        print("\n📈 COMPLEX REASONING RAPOR ÖZETİ:")
        print("-" * 60)
        print(f"Ollama - Toplam Task: {metrics['ollama']['total_tasks']:,}")
        print(f"VLLM   - Toplam Task: {metrics['vllm']['total_tasks']:,}")
        print(f"Ollama - Başarılı Task: {metrics['ollama']['total_successful']:,}")
        print(f"VLLM   - Başarılı Task: {metrics['vllm']['total_successful']:,}")
        print(f"Ollama - Ortalama Doğruluk: {metrics['ollama']['avg_accuracy']:.1f}%")
        print(f"VLLM   - Ortalama Doğruluk: {metrics['vllm']['avg_accuracy']:.1f}%")
        print(f"Ollama - Ortalama Throughput: {metrics['ollama']['avg_throughput']:.3f} task/s")
        print(f"VLLM   - Ortalama Throughput: {metrics['vllm']['avg_throughput']:.3f} task/s")
        print(f"Ollama - Ortalama Hata Oranı: {metrics['ollama']['avg_error_rate']:.1f}%")
        print(f"VLLM   - Ortalama Hata Oranı: {metrics['vllm']['avg_error_rate']:.1f}%")
        
        # Kazanan platformu belirle
        print("\n🏆 KAZANANLAR:")
        print("-" * 30)
        if metrics['vllm']['avg_accuracy'] > metrics['ollama']['avg_accuracy']:
            print("🎯 Doğruluk Kazananı: VLLM")
        else:
            print("🎯 Doğruluk Kazananı: Ollama")
            
        if metrics['vllm']['avg_throughput'] > metrics['ollama']['avg_throughput']:
            print("🚀 Throughput Kazananı: VLLM")
        else:
            print("🚀 Throughput Kazananı: Ollama")
            
        if metrics['vllm']['avg_error_rate'] < metrics['ollama']['avg_error_rate']:
            print("🛡️  Güvenilirlik Kazananı: VLLM")
        else:
            print("🛡️  Güvenilirlik Kazananı: Ollama")
            
    except Exception as e:
        print(f"❌ Hata: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
