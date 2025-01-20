from comet import download_model, load_from_checkpoint
import pandas as pd
import numpy as np


device = [0]
lang = 'de'
df = pd.read_csv(f'/home/raychen/20240729/baseline_acl2025/rain/20/zh-{lang}_paragraph_20.csv') # for training-time
src = df['zh'].replace('</s>', '')
mt = df['rain'].replace('</s>', '')
ref = df[lang].replace('</s>', '')

data = [
    {
        "src": src[i],
        "mt": mt[i],
        "ref": ref[i]
    } for i in range(len(df))
]
print(f'{len(data)=}')
data2 = [
    {
        "src": src[i],
        "mt": mt[i],
    } for i in range(len(df))
]
print(f'{len(data2)=}')

model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)
model_output = model.predict(data, batch_size=8, gpus=1, devices=device)

model2_path = download_model("Unbabel/wmt22-cometkiwi-da")
model2 = load_from_checkpoint(model2_path)
model2_output = model2.predict(data2, batch_size=8, gpus=1, devices=device)

print(f'{lang}_ref-based_{np.mean(model_output.scores):.2f}±{np.std(model_output.scores):.2f}')
print(f'{lang}_ref-free_{np.mean(model2_output.scores):.2f}±{np.std(model2_output.scores):.2f}')