import pandas as pd
from sklearn.model_selection import train_test_split
import imagehash
import numpy as np
from PIL import Image, UnidentifiedImageError

#Archivo que contiene funciones auxiliares tomadas el archivo lab3_EDA

def son_realmente_duplicados(ruta1, ruta2, tam=(64, 64), umbral_mse=15):
    """Verificación adicional: compara los píxeles de verdad, no solo el hash."""
    img1 = np.array(Image.open(ruta1).convert("RGB").resize(tam), dtype=np.float64)
    img2 = np.array(Image.open(ruta2).convert("RGB").resize(tam), dtype=np.float64)
    mse = np.mean((img1 - img2) ** 2)
    return mse < umbral_mse, mse

def marcar_duplicado_verificado(grupo_df, umbral_mse=15):
    """Dentro de un grupo con mismo phash, solo confirma como duplicado si el MSE es bajo."""
    indices_a_eliminar = []
    rutas = grupo_df["ruta"].tolist()
    indices = grupo_df.index.tolist()
    referencia = rutas[0]
    for idx, ruta in zip(indices[1:], rutas[1:]):
        es_dup, _ = son_realmente_duplicados(referencia, ruta, umbral_mse=umbral_mse)
        if es_dup:
            indices_a_eliminar.append(idx)
    return indices_a_eliminar

def calcular_hash(ruta, hash_size=16):
    try:
        img = Image.open(ruta)
        # hash_size mayor = huella más detallada = menos colisiones falsas
        return str(imagehash.phash(img, hash_size=hash_size))
    except Exception:
        return None

def dividir_train_val_test(
    df: pd.DataFrame, 
    col_clase: str = "clase", 
    semilla: int = 42, 
    test_size: float = 0.30, 
    val_size_of_temp: float = 0.50,
    verbose: bool = True
):
    """
    Divide un DataFrame en conjuntos de Train (70%), Validation (15%) y Test (15%) 
    manteniendo la proporción de clases (estratificación).
    """
    train_df, temp_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df[col_clase], 
        random_state=semilla
    )
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=val_size_of_temp, 
        stratify=temp_df[col_clase], 
        random_state=semilla
    )
    
    if verbose:
        total = len(df)
        print(f"Train: {len(train_df)} ({len(train_df)/total:.1%}) | "
              f"Val: {len(val_df)} ({len(val_df)/total:.1%}) | "
              f"Test: {len(test_df)} ({len(test_df)/total:.1%})")
        
    return train_df, val_df, test_df