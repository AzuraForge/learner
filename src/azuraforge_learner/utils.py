# learner/src/azuraforge_learner/utils.py

import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def set_professional_style():
    """Matplotlib için profesyonel bir stil ayarlar."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': 'DejaVu Sans',
        'figure.figsize': (12, 7),
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'grid.color': '#dddddd'
    })

def plot_loss_history(history: Dict[str, List[float]], save_path: str):
    """Eğitim ve doğrulama kaybını çizer ve kaydeder."""
    set_professional_style()
    plt.figure()
    plt.plot(history['loss'], label='Eğitim Kaybı')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Doğrulama Kaybı')
    plt.title('Model Öğrenme Eğrisi')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp (Loss)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_prediction_comparison(y_true: np.ndarray, y_pred: np.ndarray, time_index: pd.Index, save_path: str, y_label: str):
    """Gerçek ve tahmin edilen değerleri zaman serisi olarak çizer."""
    set_professional_style()
    plt.figure()
    plt.plot(time_index, y_true, label='Gerçek Değerler', marker='.')
    plt.plot(time_index, y_pred, label='Tahmin Edilen Değerler', linestyle='--')
    plt.title('Tahmin vs Gerçek Değerler')
    plt.xlabel('Tarih')
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_regression_report(results: Dict[str, Any], config: Dict[str, Any]):
    """
    Regresyon deneyi sonuçlarından bir Markdown raporu oluşturur.
    """
    experiment_dir = config['experiment_dir']
    report_name = config.get('pipeline_name', 'Bilinmeyen Deney')
    
    img_dir = os.path.join(experiment_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    report_path = os.path.join(experiment_dir, "report.md")
    logging.info(f"Regresyon raporu oluşturuluyor: {report_path}")

    # Grafikleri oluştur
    loss_img_path = os.path.join(img_dir, "loss_history.png")
    plot_loss_history(results['history'], save_path=loss_img_path)

    comparison_img_path = os.path.join(img_dir, "prediction_comparison.png")
    if 'y_true' in results and 'y_pred' in results and 'time_index' in results:
        plot_prediction_comparison(
            y_true=results['y_true'],
            y_pred=results['y_pred'],
            time_index=results['time_index'],
            save_path=comparison_img_path,
            y_label=results.get('y_label', 'Değer')
        )

    # Metrikleri al
    metrics = results.get('metrics', {})
    r2 = metrics.get('r2_score', 'N/A')
    mae = metrics.get('mae', 'N/A')
    
    # Raporu yaz
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Regresyon Analiz Raporu: {report_name}\n")
        f.write(f"**Rapor Tarihi:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 1. Performans Özeti\n")
        f.write(f"- **R² Skoru:** `{r2:.4f}`\n")
        f.write(f"- **Ortalama Mutlak Hata (MAE):** `{mae:.4f}`\n\n")

        f.write("## 2. Tahmin Karşılaştırması\n")
        f.write("Aşağıdaki grafik, modelin test seti üzerindeki tahminlerini (turuncu) gerçek değerlerle (mavi) karşılaştırır.\n\n")
        f.write("![Tahmin Karşılaştırma Grafiği](images/prediction_comparison.png)\n\n")
        
        f.write("## 3. Eğitim Süreci\n")
        f.write("Bu grafik, modelin eğitim sırasındaki kayıp değerinin epoch'lara göre değişimini gösterir.\n\n")
        f.write("![Eğitim Kaybı](images/loss_history.png)\n\n")

        f.write("## 4. Deney Konfigürasyonu\n")
        f.write("```json\n")
        f.write(json.dumps(config, indent=4))
        f.write("\n```\n")