import json
from pathlib import Path
path = Path(r'D:/Documentos/Trabajos/Laboratorios/NOTEBOOK_ESTADISTICA_CRYPTO/N1_PRONOSTICO_PROPHET.ipynb')
data = json.loads(path.read_text(encoding='utf-8'))
modified = False
for cell in data['cells']:
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell['source'])
    if src.startswith("hourly_data = yf.download(ticker, period='6mo', interval='60m', progress=False)\n") and 'hourly_data = hourly_data[[\'Close\']].dropna().reset_index()' in src:
        lines = cell['source']
        new_lines = []
        for line in lines:
            if line.startswith("hourly_data = yf.download"):
                new_lines.append(line)
                new_lines.append("if isinstance(hourly_data.columns, pd.MultiIndex):\n")
                new_lines.append("    hourly_data.columns = hourly_data.columns.get_level_values(-1)\n")
                continue
            if line.startswith("hourly_data = hourly_data[['Close']].dropna().reset_index()"):
                new_lines.append(line)
                new_lines.append("hourly_data = hourly_data.rename(columns={'Datetime': 'ds', 'Date': 'ds', 'Close': 'y'})\n")
                new_lines.append("hourly_data['ds'] = pd.to_datetime(hourly_data['ds'])\n")
                continue
            if line.startswith("hourly_data = hourly_data.rename(columns={'Datetime': 'ds', 'Date': 'ds', 'Close': 'y'})"):
                continue
            if line.startswith("hourly_data['ds'] = pd.to_datetime(hourly_data['ds'])"):
                continue
            new_lines.append(line)
        cell['source'] = new_lines
        modified = True
if modified:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
    print('hourly updated')
else:
    print('no changes applied')
