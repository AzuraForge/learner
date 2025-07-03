# learner/src/azuraforge_learner/reporting.py

import os
import logging
import json # EKSİK OLAN IMPORT EKLENDİ
from datetime import datetime
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def set_professional_style():
    """Matplotlib için profesyonel bir stil ayarlar."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.family': 'sans-serif', 'font.sans-serif': 'DejaVu Sans',
            'figure.figsize': (12, 7), 'axes.labelweight': 'bold',
            'axes.titleweight': 'bold', 'grid.color': '#dddddd'
        })
    except Exception as e:
        logging.warning(f"Matplotlib stili yüklenemedi: {e}. Varsayılan kullanılacak.")

def plot_loss_history(history: Dict[str, List[float]], save_path: str):
    set_professional_style()
    fig, ax = plt.subplots()
    ax.plot(history.get('loss', []), label='Eğitim Kaybı')
    if 'val_loss' in history:
        ax.plot(history['val_loss'], label='Doğrulama Kaybı')
    ax.set_title('Model Öğrenme Eğrisi')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Kayıp (Loss)')
    ax.legend()
    ax.grid(True)
    fig.savefig(save_path)
    plt.close(fig)

def plot_prediction_comparison(y_true: np.ndarray, y_pred: np.ndarray, time_index: pd.Index, save_path: str, y_label: str):
    set_professional_style()
    fig, ax = plt.subplots()
    ax.plot(time_index, y_true, label='Gerçek Değerler', marker='.', markersize=4, linestyle='-')
    ax.plot(time_index, y_pred, label='Tahmin Edilen Değerler', linestyle='--')
    ax.set_title('Tahmin vs Gerçek Değerler')
    ax.set_xlabel('Tarih')
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

# ... (plot_prediction_comparison fonksiyonunun sonu)
import itertools # Dosyanın başında import edildiğinden emin olun

def plot_confusion_matrix(cm, classes, save_path, normalize=False, title='Confusion matrix', cmap=plt.cm.Greens):
    """
    Bu fonksiyon, confusion matrix'i çizer ve kaydeder.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    set_professional_style()
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im)
    
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('Gerçek Etiket (True Label)')
    ax.set_xlabel('Tahmin Edilen Etiket (Predicted Label)')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def generate_classification_report(results: Dict[str, Any], config: Dict[str, Any], class_names: List[str]):
    experiment_dir = config.get('experiment_dir')
    if not experiment_dir:
        logging.error("Rapor oluşturmak için 'experiment_dir' konfigürasyonda bulunamadı.")
        return
        
    report_name = config.get('pipeline_name', 'Bilinmeyen Deney')
    img_dir = os.path.join(experiment_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    report_path = os.path.join(experiment_dir, "report.md")
    logging.info(f"Sınıflandırma raporu oluşturuluyor: {report_path}")

    # Confusion Matrix'i çizdir
    cm_img_path = os.path.join(img_dir, "confusion_matrix.png")
    if 'confusion_matrix' in results:
        cm = np.array(results['confusion_matrix'])
        plot_confusion_matrix(cm, classes=class_names, save_path=cm_img_path, title='Confusion Matrix')

    metrics = results.get('metrics', {})
    accuracy = metrics.get('accuracy')
    class_report = metrics.get('classification_report', {})

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Sınıflandırma Analiz Raporu: {report_name}\n\n")
        f.write(f"**Rapor Tarihi:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 1. Performans Özeti\n\n")
        if accuracy is not None:
            f.write(f"- **Genel Doğruluk (Accuracy):** `{accuracy:.4f}`\n\n")
        
        f.write("### Sınıf Bazında Metrikler\n\n")
        f.write("| Sınıf | Precision | Recall | F1-Score | Support |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|\n")
        for class_name, report_metrics in class_report.items():
            if isinstance(report_metrics, dict):
                p = report_metrics.get('precision', 0)
                r = report_metrics.get('recall', 0)
                f1 = report_metrics.get('f1-score', 0)
                s = report_metrics.get('support', 0)
                f.write(f"| {class_name} | {p:.2f} | {r:.2f} | {f1:.2f} | {s} |\n")
        f.write("\n")
        
        f.write("## 2. Karmaşıklık Matrisi (Confusion Matrix)\n\n")
        f.write("Bu matris, modelin hangi sınıfları birbiriyle karıştırdığını gösterir.\n\n")
        if os.path.exists(cm_img_path):
            f.write(f"![Karmaşıklık Matrisi](images/confusion_matrix.png)\n\n")

        f.write("## 3. Deney Konfigürasyonu\n\n")
        f.write("```json\n")
        f.write(json.dumps(config, indent=4, default=str))
        f.write("\n```\n")    

def generate_regression_report(results: Dict[str, Any], config: Dict[str, Any]):
    experiment_dir = config.get('experiment_dir')
    if not experiment_dir:
        logging.error("Rapor oluşturmak için 'experiment_dir' konfigürasyonda bulunamadı.")
        return
        
    report_name = config.get('pipeline_name', 'Bilinmeyen Deney')
    
    img_dir = os.path.join(experiment_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    report_path = os.path.join(experiment_dir, "report.md")
    logging.info(f"Regresyon raporu oluşturuluyor: {report_path}")

    loss_img_path = os.path.join(img_dir, "loss_history.png")
    if 'history' in results and results['history'].get('loss'):
        plot_loss_history(results['history'], save_path=loss_img_path)

    comparison_img_path = os.path.join(img_dir, "prediction_comparison.png")
    if 'y_true' in results and 'y_pred' in results and 'time_index' in results:
        plot_prediction_comparison(
            y_true=np.asarray(results['y_true']), y_pred=np.asarray(results['y_pred']),
            time_index=results['time_index'], save_path=comparison_img_path,
            y_label=results.get('y_label', 'Değer')
        )

    metrics = results.get('metrics', {})
    r2 = metrics.get('r2_score')
    mae = metrics.get('mae')
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Regresyon Analiz Raporu: {report_name}\n\n")
        f.write(f"**Rapor Tarihi:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 1. Performans Özeti\n\n")
        if r2 is not None:
            f.write(f"- **R² Skoru:** `{r2:.4f}`\n")
        if mae is not None:
            f.write(f"- **Ortalama Mutlak Hata (MAE):** `{mae:.4f}`\n\n")

        f.write("## 2. Tahmin Karşılaştırması\n\n")
        f.write("Aşağıdaki grafik, modelin test seti üzerindeki tahminlerini (turuncu) gerçek değerlerle (mavi) karşılaştırır.\n\n")
        if os.path.exists(comparison_img_path):
            f.write(f"![Tahmin Karşılaştırma Grafiği](images/{os.path.basename(comparison_img_path)})\n\n")
        
        f.write("## 3. Eğitim Süreci\n\n")
        f.write("Bu grafik, modelin eğitim sırasındaki kayıp değerinin epoch'lara göre değişimini gösterir.\n\n")
        if os.path.exists(loss_img_path):
            f.write(f"![Eğitim Kaybı](images/{os.path.basename(loss_img_path)})\n\n")

        f.write("## 4. Deney Konfigürasyonu\n\n")
        f.write("```json\n")
        f.write(json.dumps(config, indent=4, default=str))
        f.write("\n```\n")