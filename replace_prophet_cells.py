import json
from pathlib import Path
path = Path(r'D:/Documentos/Trabajos/Laboratorios/NOTEBOOK_ESTADISTICA_CRYPTO/N1_PRONOSTICO_PROPHET.ipynb')
data = json.loads(path.read_text(encoding='utf-8'))
new_daily = [
"ticker = 'GC=F'\n",
"daily_data = yf.download(ticker, period='6mo', interval='1d', progress=False)\n",
"if isinstance(daily_data.columns, pd.MultiIndex):\n",
"    daily_data.columns = daily_data.columns.droplevel('Ticker')\n",
"daily_data = daily_data[['Close']].dropna().reset_index()\n",
"daily_data = daily_data.rename(columns={'Date': 'ds', 'Close': 'y'})\n",
"daily_data['ds'] = pd.to_datetime(daily_data['ds'])\n",
"print('Datos diarios cargados:', daily_data.shape)\n",
"daily_data.head()\n"
]
new_hourly = [
"hourly_data = yf.download(ticker, period='6mo', interval='60m', progress=False)\n",
"if hourly_data.empty:\n",
"    hourly_data = yf.download(ticker, period='60d', interval='60m', progress=False)\n",
"if isinstance(hourly_data.columns, pd.MultiIndex):\n",
"    hourly_data.columns = hourly_data.columns.droplevel('Ticker')\n",
"hourly_data = hourly_data[['Close']].dropna().reset_index()\n",
"hourly_data = hourly_data.rename(columns={'Datetime': 'ds', 'Date': 'ds', 'Close': 'y'})\n",
"hourly_data['ds'] = pd.to_datetime(hourly_data['ds'])\n",
"print('Datos horarios cargados:', hourly_data.shape)\n",
"hourly_data.head()\n"
]
for cell in data['cells']:
    if cell.get('cell_type') != 'code':
        continue
    source = ''.join(cell['source'])
    if source.startswith("ticker = 'GC=F'\n") and 'daily_data = yf.download' in source:
        cell['source'] = new_daily
    if source.startswith("hourly_data = yf.download(ticker, period='6mo', interval='60m', progress=False)\n"):
        cell['source'] = new_hourly
path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
print('updated')
