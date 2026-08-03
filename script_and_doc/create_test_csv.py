import pandas as pd
from code.utils.config import FEATURE_SPECS

test_data = {}
for f in FEATURE_SPECS:
    if f.order and len(f.order) > 0:
        test_data[f.name] = f.order[0]
    else:
        test_data[f.name] = f.default

df = pd.DataFrame([test_data])
df.to_csv('test_upload.csv', index=False)
print('Test CSV created with valid values')
print(df)
