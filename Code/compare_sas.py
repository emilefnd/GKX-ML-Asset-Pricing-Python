import numpy as np
import pandas as pd

PY_FILE = 'rpsdata_python.parquet'
SAS_FILE = 'SAS/rpsdata_rfs.parquet'
END = '2016-12-31'  # daily CRSP was only pulled to 2016 in the Python port

py = pd.read_parquet(PY_FILE)
sas = pd.read_parquet(SAS_FILE)
sas.columns = sas.columns.str.lower()  # SAS names are case-insensitive

# match on permno and calendar month (both use CRSP month-end dates, this avoids any day mismatch)
for df in (py, sas):
    df['permno'] = df['permno'].astype('int64')
    df['ym'] = pd.to_datetime(df['date']).dt.to_period('M')
py = py[py['date'] <= END]
sas = sas[pd.to_datetime(sas['date']) <= END]

print(f'rows  python: {len(py):,}   sas: {len(sas):,}')
print(f'duplicate (permno, month)  python: {py.duplicated(["permno", "ym"]).sum():,}   '
      f'sas: {sas.duplicated(["permno", "ym"]).sum():,}')

m = py.merge(sas, on=['permno', 'ym'], how='outer', suffixes=('_py', '_sas'), indicator=True)
print(m['_merge'].value_counts().rename({'left_only': 'python only', 'right_only': 'sas only'}).to_string(), '\n')
both = m[m['_merge'] == 'both']

ids = {'permno', 'date', 'ym', 'gvkey', 'gvkey_x', 'gvkey_y', 'datadate', 'rdq', 'fyear', 'sic2', 'count'}
common = sorted((set(py.columns) & set(sas.columns)) - ids)
print('only in python:', sorted(set(py.columns) - set(sas.columns) - ids))
print('only in sas:   ', sorted(set(sas.columns) - set(py.columns) - ids), '\n')

rows = []
for v in common:
    a = pd.to_numeric(both[f'{v}_py'], errors='coerce').astype('float64')
    b = pd.to_numeric(both[f'{v}_sas'], errors='coerce').astype('float64')
    n_inf = int(np.isinf(a).sum())  # SAS turns x/0 and log(0) into missing, so these should be 0
    a, b = a.replace([np.inf, -np.inf], np.nan), b.replace([np.inf, -np.inf], np.nan)
    ok = a.notna() & b.notna()
    close = np.isclose(a[ok], b[ok], rtol=1e-5, atol=1e-6)  # float32 rounding tolerance
    rows.append({
        'var': v,
        'n_py': int(a.notna().sum()),
        'n_sas': int(b.notna().sum()),
        'only_py': int((a.notna() & b.isna()).sum()),
        'only_sas': int((a.isna() & b.notna()).sum()),
        'inf_py': n_inf,
        'corr': a[ok].corr(b[ok]) if ok.sum() > 2 else np.nan,
        'pct_equal': 100 * close.mean() if ok.any() else np.nan,
    })

res = pd.DataFrame(rows).sort_values('pct_equal')
pd.set_option('display.width', 200, 'display.max_rows', 200)
print(res.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
res.to_csv('compare_sas_results.csv', index=False)
print('\nsaved: compare_sas_results.csv')
