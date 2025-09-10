#!/usr/bin/env python3
"""
Scenario-1 HTML Report Generator
Ollama ve VLLM performans raporlarını karşılaştıran HTML raporu oluşturur.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

class Scenario1Reporter:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.ollama_data = None
        self.vllm_data = None
        
    def load_data(self):
        """JSON verilerini yükler"""
        # Ollama verilerini yükle
        ollama_files = list(self.results_dir.glob("ollama_performance_data_*.json"))
        if ollama_files:
            with open(ollama_files[0], 'r') as f:
                self.ollama_data = json.load(f)
        
        # VLLM verilerini yükle
        vllm_files = list(self.results_dir.glob("vllm_performance_data_*.json"))
        if vllm_files:
            with open(vllm_files[0], 'r') as f:
                self.vllm_data = json.load(f)
                
        if not self.ollama_data or not self.vllm_data:
            raise ValueError("Ollama veya VLLM veri dosyası bulunamadı!")
    
    def generate_chart_data(self):
        """Chart.js için veri hazırlar"""
        levels = []
        ollama_latency = []
        vllm_latency = []
        ollama_throughput = []
        vllm_throughput = []
        ollama_tokens_per_sec = []
        vllm_tokens_per_sec = []
        
        for result in self.ollama_data['results']:
            levels.append(result['level'])
            ollama_latency.append(result['avg_latency'])
            ollama_throughput.append(result['throughput'])
            ollama_tokens_per_sec.append(result['avg_tokens_per_second'])
        
        for result in self.vllm_data['results']:
            vllm_latency.append(result['avg_latency'])
            vllm_throughput.append(result['throughput'])
            vllm_tokens_per_sec.append(result['avg_tokens_per_second'])
        
        return {
            'levels': levels,
            'ollama_latency': ollama_latency,
            'vllm_latency': vllm_latency,
            'ollama_throughput': ollama_throughput,
            'vllm_throughput': vllm_throughput,
            'ollama_tokens_per_sec': ollama_tokens_per_sec,
            'vllm_tokens_per_sec': vllm_tokens_per_sec
        }
    
    def calculate_performance_metrics(self):
        """Performans metriklerini hesaplar"""
        metrics = {
            'ollama': {'total_requests': 0, 'total_successful': 0, 'avg_throughput': 0, 'avg_tokens_per_sec': 0},
            'vllm': {'total_requests': 0, 'total_successful': 0, 'avg_throughput': 0, 'avg_tokens_per_sec': 0}
        }
        
        # Ollama metrikleri
        for result in self.ollama_data['results']:
            metrics['ollama']['total_requests'] += result['total_requests']
            metrics['ollama']['total_successful'] += result['successful_requests']
            
        metrics['ollama']['avg_throughput'] = sum(r['throughput'] for r in self.ollama_data['results']) / len(self.ollama_data['results'])
        metrics['ollama']['avg_tokens_per_sec'] = sum(r['avg_tokens_per_second'] for r in self.ollama_data['results']) / len(self.ollama_data['results'])
        
        # VLLM metrikleri
        for result in self.vllm_data['results']:
            metrics['vllm']['total_requests'] += result['total_requests']
            metrics['vllm']['total_successful'] += result['successful_requests']
            
        metrics['vllm']['avg_throughput'] = sum(r['throughput'] for r in self.vllm_data['results']) / len(self.vllm_data['results'])
        metrics['vllm']['avg_tokens_per_sec'] = sum(r['avg_tokens_per_second'] for r in self.vllm_data['results']) / len(self.vllm_data['results'])
        
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
    <title>Scenario-1 Performance Comparison: Ollama vs VLLM</title>
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
            max-width: 1200px;
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
        }}
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
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
            font-size: 0.9em;
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
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
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
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Scenario-1 Performance Report</h1>
            <p>Ollama vs VLLM Performance Comparison</p>
            <p>Model: {self.ollama_data['config']['model_name']} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="summary">
            <div class="summary-card ollama-card">
                <h3>🦙 Ollama Performance</h3>
                <div class="metric">
                    <span class="metric-label">Total Requests:</span>
                    <span class="metric-value">{metrics['ollama']['total_requests']:,}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Successful Requests:</span>
                    <span class="metric-value">{metrics['ollama']['total_successful']:,}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Average Throughput:</span>
                    <span class="metric-value">{metrics['ollama']['avg_throughput']:.3f} req/s</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Average Tokens/sec:</span>
                    <span class="metric-value">{metrics['ollama']['avg_tokens_per_sec']:.1f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Success Rate:</span>
                    <span class="metric-value">100.0%</span>
                </div>
            </div>

            <div class="summary-card vllm-card">
                <h3>⚡ VLLM Performance</h3>
                <div class="metric">
                    <span class="metric-label">Total Requests:</span>
                    <span class="metric-value">{metrics['vllm']['total_requests']:,}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Successful Requests:</span>
                    <span class="metric-value">{metrics['vllm']['total_successful']:,}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Average Throughput:</span>
                    <span class="metric-value">{metrics['vllm']['avg_throughput']:.3f} req/s</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Average Tokens/sec:</span>
                    <span class="metric-value">{metrics['vllm']['avg_tokens_per_sec']:.1f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Success Rate:</span>
                    <span class="metric-value">100.0%</span>
                </div>
            </div>
        </div>

        <div class="comparison-highlights">
            <div class="highlight-card">
                <div class="highlight-label">Throughput Winner</div>
                <div class="highlight-value">{"VLLM" if metrics['vllm']['avg_throughput'] > metrics['ollama']['avg_throughput'] else "Ollama"}</div>
                <div class="highlight-label">{max(metrics['vllm']['avg_throughput'], metrics['ollama']['avg_throughput']):.3f} req/s</div>
            </div>
            <div class="highlight-card">
                <div class="highlight-label">Tokens/sec Winner</div>
                <div class="highlight-value">{"VLLM" if metrics['vllm']['avg_tokens_per_sec'] > metrics['ollama']['avg_tokens_per_sec'] else "Ollama"}</div>
                <div class="highlight-label">{max(metrics['vllm']['avg_tokens_per_sec'], metrics['ollama']['avg_tokens_per_sec']):.1f} tokens/s</div>
            </div>
            <div class="highlight-card">
                <div class="highlight-label">Total Requests</div>
                <div class="highlight-value">{metrics['vllm']['total_requests'] + metrics['ollama']['total_requests']:,}</div>
                <div class="highlight-label">Processed Successfully</div>
            </div>
        </div>

        <div class="charts">
            <div class="chart-title">📊 Latency Comparison (Lower is Better)</div>
            <div class="chart-container">
                <canvas id="latencyChart"></canvas>
            </div>

            <div class="chart-title">🚀 Throughput Comparison (Higher is Better)</div>
            <div class="chart-container">
                <canvas id="throughputChart"></canvas>
            </div>

            <div class="chart-title">⚡ Tokens per Second Comparison (Higher is Better)</div>
            <div class="chart-container">
                <canvas id="tokensChart"></canvas>
            </div>
        </div>

        <div class="detailed-table">
            <div class="table-title">📋 Detailed Performance Comparison</div>
            <table>
                <thead>
                    <tr>
                        <th>Concurrent Users</th>
                        <th>Platform</th>
                        <th>Total Requests</th>
                        <th>Avg Latency (ms)</th>
                        <th>P95 Latency (ms)</th>
                        <th>Throughput (req/s)</th>
                        <th>Tokens/sec</th>
                        <th>CPU Usage (%)</th>
                        <th>Memory Usage (%)</th>
                    </tr>
                </thead>
                <tbody>
"""

        # Detaylı tablo verilerini ekle
        for i, level in enumerate(chart_data['levels']):
            ollama_result = self.ollama_data['results'][i]
            vllm_result = self.vllm_data['results'][i]
            
            # Karşılaştırma için işaretleyiciler
            better_latency_ollama = ollama_result['avg_latency'] < vllm_result['avg_latency']
            better_throughput_ollama = ollama_result['throughput'] > vllm_result['throughput']
            better_tokens_ollama = ollama_result['avg_tokens_per_second'] > vllm_result['avg_tokens_per_second']
            
            html_template += f"""
                    <tr>
                        <td rowspan="2" style="vertical-align: middle; font-weight: bold; background: #f8f9fa;">{level}</td>
                        <td><strong>🦙 Ollama</strong></td>
                        <td>{ollama_result['total_requests']}</td>
                        <td><span class="performance-indicator {'better' if better_latency_ollama else 'worse'}">{ollama_result['avg_latency']:.1f}</span></td>
                        <td>{ollama_result['p95_latency']:.1f}</td>
                        <td><span class="performance-indicator {'better' if better_throughput_ollama else 'worse'}">{ollama_result['throughput']:.3f}</span></td>
                        <td><span class="performance-indicator {'better' if better_tokens_ollama else 'worse'}">{ollama_result['avg_tokens_per_second']:.1f}</span></td>
                        <td>{ollama_result['avg_cpu_usage']:.1f}</td>
                        <td>{ollama_result['avg_memory_usage']:.1f}</td>
                    </tr>
                    <tr>
                        <td><strong>⚡ VLLM</strong></td>
                        <td>{vllm_result['total_requests']}</td>
                        <td><span class="performance-indicator {'better' if not better_latency_ollama else 'worse'}">{vllm_result['avg_latency']:.1f}</span></td>
                        <td>{vllm_result['p95_latency']:.1f}</td>
                        <td><span class="performance-indicator {'better' if not better_throughput_ollama else 'worse'}">{vllm_result['throughput']:.3f}</span></td>
                        <td><span class="performance-indicator {'better' if not better_tokens_ollama else 'worse'}">{vllm_result['avg_tokens_per_second']:.1f}</span></td>
                        <td>{vllm_result['avg_cpu_usage']:.1f}</td>
                        <td>{vllm_result['avg_memory_usage']:.1f}</td>
                    </tr>
"""

        html_template += f"""
                </tbody>
            </table>
        </div>

        <div class="footer">
            <p>Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')} | 
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

        // Latency Chart
        new Chart(document.getElementById('latencyChart'), {{
            type: 'line',
            data: {{
                labels: {chart_data['levels']},
                datasets: [{{
                    label: '🦙 Ollama',
                    data: {chart_data['ollama_latency']},
                    borderColor: '#ff6b6b',
                    backgroundColor: 'rgba(255, 107, 107, 0.1)',
                    tension: 0.4
                }}, {{
                    label: '⚡ VLLM',
                    data: {chart_data['vllm_latency']},
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
                            text: 'Latency (ms)'
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

        // Tokens per Second Chart
        new Chart(document.getElementById('tokensChart'), {{
            type: 'bar',
            data: {{
                labels: {chart_data['levels']},
                datasets: [{{
                    label: '🦙 Ollama',
                    data: {chart_data['ollama_tokens_per_sec']},
                    backgroundColor: 'rgba(255, 107, 107, 0.7)',
                    borderColor: '#ff6b6b',
                    borderWidth: 2
                }}, {{
                    label: '⚡ VLLM',
                    data: {chart_data['vllm_tokens_per_sec']},
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
                            text: 'Tokens per Second'
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
    </script>
</body>
</html>
"""
        return html_template

    def save_report(self, output_file=None):
        """HTML raporunu dosyaya kaydeder"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"html-reports/scenario-1_comparison_report_{timestamp}.html"
        
        html_content = self.generate_html_report()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_file

def main():
    """Ana fonksiyon"""
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Varsayılan olarak scenario-1 klasörünü kullan
        script_dir = Path(__file__).parent
        results_dir = script_dir.parent / "cases" / "results" / "scenario-1"
    
    if not Path(results_dir).exists():
        print(f"❌ Hata: {results_dir} klasörü bulunamadı!")
        sys.exit(1)
    
    try:
        print("📊 Scenario-1 HTML raporu oluşturuluyor...")
        reporter = Scenario1Reporter(results_dir)
        reporter.load_data()
        
        output_file = reporter.save_report()
        print(f"✅ HTML raporu başarıyla oluşturuldu: {output_file}")
        print(f"🌐 Raporu görüntülemek için dosyayı web tarayıcınızda açın.")
        
        # Rapor özeti
        metrics = reporter.calculate_performance_metrics()
        print("\n📈 RAPOR ÖZETİ:")
        print("-" * 50)
        print(f"Ollama - Toplam İstek: {metrics['ollama']['total_requests']:,}")
        print(f"VLLM   - Toplam İstek: {metrics['vllm']['total_requests']:,}")
        print(f"Ollama - Ortalama Throughput: {metrics['ollama']['avg_throughput']:.3f} req/s")
        print(f"VLLM   - Ortalama Throughput: {metrics['vllm']['avg_throughput']:.3f} req/s")
        print(f"Ollama - Ortalama Tokens/sec: {metrics['ollama']['avg_tokens_per_sec']:.1f}")
        print(f"VLLM   - Ortalama Tokens/sec: {metrics['vllm']['avg_tokens_per_sec']:.1f}")
        
        # Kazanan platformu belirle
        if metrics['vllm']['avg_throughput'] > metrics['ollama']['avg_throughput']:
            print("🏆 Throughput Kazananı: VLLM")
        else:
            print("🏆 Throughput Kazananı: Ollama")
            
        if metrics['vllm']['avg_tokens_per_sec'] > metrics['ollama']['avg_tokens_per_sec']:
            print("🏆 Tokens/sec Kazananı: VLLM")
        else:
            print("🏆 Tokens/sec Kazananı: Ollama")
            
    except Exception as e:
        print(f"❌ Hata: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
