import os
import requests
import sys
import progressbar

from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

URL_MODEL = 'https://huggingface.co/VietAI/gpt-neo-1.3B-vietnamese-news/' \
    'resolve/main/pytorch_model.bin'
start_time = datetime.utcnow()
trained_model_path = os.path.join(os.getcwd(), 'pytorch_model.bin')

if not os.path.exists(trained_model_path):
    with requests.get(URL_MODEL, stream=True) as trained_remote_file:
        data_size = int(
            trained_remote_file.headers.get('Content-Length').strip())
        data_downloaded = 0

        # Tạo progressbar
        download_widget = [
            'Đang tải model: ',
            progressbar.Bar(
                marker=progressbar.AnimatedMarker(
                    markers='█',
                    fill='█',
                    fill_wrap='\x1b[32m{}\x1b[39m',
                    marker_wrap='\x1b[32m{}\x1b[39m',
                ),
                left="[",
                right=" ",
            ),
            progressbar.Percentage(),
            " ",
            progressbar.FileTransferSpeed(),
            "] ",
            " of {0}MB".format(int(round(data_size / 1024 / 1024, 2))),
        ]
        current_bar = progressbar.ProgressBar(
            widgets=download_widget,
            max_value=data_size,
        ).start()

        trained_remote_file.raise_for_status()
        with open(trained_model_path, 'wb') as trained_local_file:
            for chunk in trained_remote_file.iter_content(chunk_size=8192):
                data_downloaded += len(chunk)
                current_bar.update(data_downloaded)
                trained_local_file.write(chunk)

        current_bar.finish(dirty=True)

tokenizer = AutoTokenizer.from_pretrained(os.getcwd())
model = AutoModelForCausalLM.from_pretrained(
    os.getcwd(),
    low_cpu_mem_usage=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = sys.argv[1]
prompt = sys.argv[2]
while True:
    if not str(max_length).isdigit():
        print('Số ký tự phải là số nguyên (VD: 1000)')
        max_length = input('Nhập số ký tự tối đa của đoạn văn được tạo: ')
        continue
    max_length = int(max_length)
    break

print('Tạo đoạn văn cho:')
print(f'"{prompt}"')

input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(device)

gen_tokens = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=0.9,
        top_k=20,
    )

gen_text = tokenizer.batch_decode(gen_tokens)[0]
print('Đoạn văn của bạn là')
print('=' * 50)
print(gen_text)
print('=' * 50)
total_time = datetime.utcnow() - start_time
print(f'Thời gian xử lý: {total_time} giây')
