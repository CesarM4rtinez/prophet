import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# =====================================================================
# UTILIDADES DE DESCARGA Y FIN DE INTERVALO
# =====================================================================

def ajustar_rango_yfinance(start, end, interval):
    """Ajusta el rango de fechas para intervalos intradiarios limitados a 730 días."""
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    intradiario = interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]
    if intradiario:
        max_days = 730
        if end_dt - start_dt >= pd.Timedelta(days=max_days):
            nuevo_inicio = end_dt - pd.Timedelta(days=max_days - 1)
            print(
                f"Advertencia: el intervalo '{interval}' solo admite hasta {max_days} días. "
                f"Ajustando inicio a {nuevo_inicio.date()} para evitar el límite de rango."
            )
            start_dt = nuevo_inicio
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


# =====================================================================
# 1. PREPROCESAMIENTO Y ESTACIONARIEDAD (Basado en tu serie temporal)
# =====================================================================
def preparar_datos_trading(ticker, start, end, interval="1h", lookback_window=60, forecast_horizon=7):
    """
    Descarga datos intradiarios y genera la matriz de características X e Y.
    Intervalos sugeridos: '5m', '15m', '30m', '1h'.
    """
    start, end = ajustar_rango_yfinance(start, end, interval)
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if df.empty and interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]:
        # Respaldo: usar el último periodo de datos intradiarios si la fecha exacta falla.
        print(
            f"Intentando descarga alternativa con period='730d' para {ticker} en intervalo {interval}..."
        )
        df = yf.download(ticker, period="730d", interval=interval, end=end, progress=False)

    if df.empty:
        raise ValueError(
            f"No se encontraron datos para {ticker} con intervalo {interval} y rango {start} -> {end}."
        )
        
    df = df[['Close', 'Open', 'High', 'Low', 'Volume']].dropna()
    
    # Transformación Estacionaria: Rendimientos Logarítmicos (Crucial según tu test ADF)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Características adicionales (Alimentación de la Red)
    df['Volatility'] = df['Log_Return'].rolling(window=10).std()
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=10).mean()
    df = df.dropna()
    
    # Definición de Etiquetas (Target): ¿El precio subirá o bajará en los próximos N periodos?
    # Para 7 días en velas de 1h -> forecast_horizon = 7 * 24 = 168 (ajustar según mercado abierto)
    df['Future_Return'] = df['Close'].shift(-forecast_horizon) - df['Close']
    
    # Clasificación triaria: 2 = Compra (Sube), 1 = Mantener (Rango), 0 = Venta (Baja)
    # Usamos umbrales basados en la desviación estándar para evitar falsos positivos
    threshold = df['Log_Return'].std() * 2
    
    def asignar_etiqueta(ret):
        if ret > threshold: return 2
        elif ret < -threshold: return 0
        else: return 1
        
    df['Target'] = df['Future_Return'].apply(asignar_etiqueta)
    df = df.dropna()
    
    # Matriz de Características (X) y Vector Objetivo (Y)
    features = ['Log_Return', 'Volatility', 'Volume_Ratio']
    X_raw = df[features].values
    y_raw = df['Target'].values
    
    # Escalamiento robusto ante outliers del mercado financiero
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # Ventana de tiempo (Estructuración 3D para redes LSTM: [muestras, time_steps, características])
    X, y = [], []
    for i in range(len(X_scaled) - lookback_window):
        X.append(X_scaled[i : i + lookback_window])
        y.append(y_raw[i + lookback_window])
        
    return np.array(X), np.array(y), scaler

# =====================================================================
# 2. DISEÑO DE LA RED NEURONAL RECURRENTE (IA para Escáner)
# =====================================================================
def construir_modelo_lstm(input_shape, num_classes=3):
    """
    Crea una arquitectura Deep Learning adaptada a la volatilidad financiera.
    """
    model = Sequential([
        # Primera Capa LSTM con Regularización por Lote para evitar desvanecimiento de gradiente
        LSTM(64, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        
        # Segunda Capa LSTM (Aprende patrones estructurales más complejos)
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        
        # Capa Densa de Clasificación (Conceptos de Redes de tu Taller Deep Learning)
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax') # Clasificación probabilística (Compra/Venta/Hold)
    ])
    
    # Compilación con optimizador robusto y función de pérdida para categorías
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# =====================================================================
# 3. PIPELINE DE EJECUCIÓN (Entrenamiento y Escáner)
# =====================================================================
if __name__ == "__main__":
    # Configuración de Parámetros
    TICKER = "BTC-USD"  # Compatible con activos de alta volatilidad
    FECHA_INICIO = "2024-01-01"
    FECHA_FIN = "2026-05-25"
    TEMPORALIDAD = "1h" # Cambiar a "5m" si posees una API de datos históricos Premium
    
    # 1. Preparar Datos
    print("Preprocesando datos del mercado...")
    try:
        X, y, scaler = preparar_datos_trading(
            ticker=TICKER, start=FECHA_INICIO, end=FECHA_FIN, interval=TEMPORALIDAD
        )
    except ValueError as error:
        print(f"ERROR: {error}")
        print("Para intervalos intradiarios (1h, 30m, 15m, 5m), Yahoo Finance ofrece hasta 730 días de datos.")
        raise
    
    # Split Entrenamiento / Validación
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # 2. Construir IA
    modelo_scanner = construir_modelo_lstm(input_shape=(X_train.shape[1], X_train.shape[2]))
    modelo_scanner.summary()
    
    # Evitar sobreentrenamiento (Overfitting) con EarlyStopping
    callback_parada = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # 3. Entrenar el Modelo
    print("\nIniciando entrenamiento del escáner con IA...")
    historial = modelo_scanner.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=[callback_parada],
        verbose=1
    )
    
    # =====================================================================
    # 4. FUNCIÓN DEL MONITOR / ESCÁNER (Simulación en Tiempo Real)
    # =====================================================================
    def escanear_mercado(modelo, ultima_ventana_datos):
        """
        Recibe la última ventana de datos del mercado y predice la acción óptima.
        """
        # Dar formato de Batch al input (1, ventana, características)
        input_datos = np.expand_dims(ultima_ventana_datos, axis=0)
        prediccion_prob = modelo.predict(input_datos, verbose=0)
        clase_predicha = np.argmax(prediccion_prob)
        
        mapa_decisiones = {0: "ALERTA DE VENTA (SHORT)", 1: "MANTENERSE AL MARGEN (HOLD)", 2: "ALERTA DE COMPRA (LONG)"}
        return mapa_decisiones[clase_predicha], prediccion_prob[0]

    # Ejecutar escáner con la última fila disponible
    resultado_ia, probabilidades = escanear_mercado(modelo_scanner, X_val[-1])
    print(f"\n--- RESULTADO DEL MONITOR IA ({TICKER}) ---")
    print(f"Decisión sugerida a 7 días: {resultado_ia}")
    print(f"Probabilidades [Venta, Esperar, Compra]: {probabilidades}")