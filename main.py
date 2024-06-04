import os
from zlib import decompress
print(list(os.environ.keys()))
os.system('pip install gpt-2-simple pandas requests')

from requests import get

promnts_url = os.environ['SECRET_URL']

with open('compressed.bin','wb') as cb: cb.write(get(promnts_url).content)
with open('compressed.bin','rb') as f: open('prompts.csv','wb').write(decompress(f.read()))
print('prossesed prompts')
import gpt_2_simple as gpt2
import os
import pandas as pd
model_name = "124M"
if not os.path.isdir(os.path.join("models", model_name)):
	print(f"Downloading {model_name} model...")
	gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/

print('Tuning')
sess = gpt2.start_tf_sess()
gpt2.finetune(sess, 'prompts.csv', model_name=model_name, steps=1000)

# Reset the TensorFlow session before finetuning again
sess.close()

for root, dirs, files in os.walk('.'):
  for file in files:
    print(os.path.join(root, file))
