# learner/src/azuraforge_learner/caching.py

import logging
import os
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
import pandas as pd

def get_cache_filepath(cache_dir: str, context: str, params: Dict[str, Any]) -> str:
    """
    Verilen parametrelere göre deterministik bir önbellek dosya yolu oluşturur.
    Dosya adı, parametrelerin sıralı bir karmasından (hash) türetilir.
    
    Args:
        cache_dir (str): Önbellek dosyalarının saklanacağı ana dizin.
        context (str): Önbelleğin ait olduğu bağlam (örn: 'stock_predictor').
        params (Dict[str, Any]): Dosya adını oluşturmak için kullanılacak parametreler.
        
    Returns:
        str: Oluşturulan tam dosya yolu.
    """
    # Parametreleri anahtarlarına göre sıralayarak tutarlı bir string oluştur
    param_str = str(sorted(params.items()))
    # Bu string'in hash'ini alarak benzersiz ve dosya sistemi için güvenli bir kimlik oluştur
    param_hash = hashlib.md5(param_str.encode()).hexdigest()
    filename = f"{context}_{param_hash}.parquet"
    
    # Bağlama özel bir alt klasör oluşturarak karışıklığı önle
    full_cache_dir = os.path.join(cache_dir, context)
    os.makedirs(full_cache_dir, exist_ok=True)
    
    return os.path.join(full_cache_dir, filename)

def load_from_cache(filepath: str, max_age_hours: int) -> Optional[pd.DataFrame]:
    """
    Veriyi önbellekten yükler. Eğer dosya yoksa veya belirtilen süreden eskiyse
    None döner.
    
    Args:
        filepath (str): Önbellek dosyasının yolu.
        max_age_hours (int): Önbelleğin saat cinsinden maksimum geçerlilik süresi.
        
    Returns:
        Optional[pd.DataFrame]: Geçerli önbellek verisi varsa DataFrame, yoksa None.
    """
    if not os.path.exists(filepath):
        return None
        
    try:
        # Dosyanın son değiştirilme zamanını al (UTC olarak)
        mod_time = datetime.fromtimestamp(os.path.getmtime(filepath), tz=timezone.utc)
        # Eğer dosyanın yaşı, izin verilen maksimum yaştan küçükse, geçerlidir
        if (datetime.now(timezone.utc) - mod_time) < timedelta(hours=max_age_hours):
            logging.info(f"Geçerli önbellek bulundu, buradan okunuyor: {filepath}")
            return pd.read_parquet(filepath)
        else:
            logging.info(f"Önbellek süresi dolmuş: {filepath}")
            os.remove(filepath) # Süresi dolmuş dosyayı temizle
            return None
    except Exception as e:
        logging.error(f"Önbellekten okuma hatası {filepath}: {e}")
        return None

def save_to_cache(df: pd.DataFrame, filepath: str) -> None:
    """
    Verilen DataFrame'i belirtilen yola Parquet formatında kaydeder.
    
    Args:
        df (pd.DataFrame): Kaydedilecek veri.
        filepath (str): Kaydedilecek dosyanın tam yolu.
    """
    try:
        # Dosyanın kaydedileceği dizinin var olduğundan emin ol
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_parquet(filepath)
        logging.info(f"Veri önbelleğe kaydedildi: {filepath}")
    except Exception as e:
        logging.error(f"Önbelleğe yazma hatası {filepath}: {e}")