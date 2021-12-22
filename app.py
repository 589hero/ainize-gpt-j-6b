import time
import torch

from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request, jsonify, render_template
from queue import Queue, Empty
from threading import Thread

app = Flask(__name__)

print("model loading...")

# # # Model & Tokenizer loading
# tokenizer = AutoTokenizer.from_pretrained("./model")
# model = AutoModelForCausalLM.from_pretrained("./model", torch_dtype=torch.float16)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

requests_queue = Queue()    # request queue.
BATCH_SIZE = 1              # max request size.
CHECK_INTERVAL = 0.1

print("complete model loading")

def handle_requests_by_batch():
    while True:
        request_batch = []

        while not (len(request_batch) >= BATCH_SIZE):
            try:
                request_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue

            for requests in request_batch:
                try:
                    requests["output"] = make_text(requests['input'])

                except Exception as e:
                    requests["output"] = e


handler = Thread(target=handle_requests_by_batch).start()


def make_text(request_input: List) -> Dict:
    return "result"
    try:
        text, max_length, temperature, top_p, repetition_penalty = request_input[0], request_input[1], request_input[2], request_input[3], request_input[4]

        input_ids = tokenizer.encode(text, return_tensors='pt')
        input_ids = input_ids.to(device)
        min_length = len(input_ids.tolist()[0])
        max_length = max_length if max_length > 0 else 1
        max_length += min_length
        gen_ids = model.generate(input_ids,
                                 max_length=max_length,
                                 temperature=temperature,
                                 do_sample=True,
                                 repetition_penalty=repetition_penalty,
                                 top_p=top_p,
                                 top_k=50)
        result = dict()

        for idx, sample_output in enumerate(gen_ids):
            result[0] = tokenizer.decode(sample_output.tolist(), skip_special_tokens=True)
        return result

    except Exception as e:
        print('Error occur in script generating!', e)
        return jsonify({'Error': e}), 500


@app.route('/generate', methods=['POST'])
def generate():
    if requests_queue.qsize() > BATCH_SIZE:
        return jsonify({'Error': 'Too Many Requests. Please try again later'}), 429

    try:
        args = []
        text = request.form['text']
        length = int(request.form.get('length', 150))
        temperature = float(request.form.get('temperature', 0.9))
        top_p = float(request.form.get('top_p', 0.95))
        repetition_penalty = float(request.form.get('repetition_penalty', 0.8))

        args.append(text)
        args.append(length)
        args.append(temperature)
        args.append(top_p)
        args.append(repetition_penalty)

        print(args)

    except Exception as e:
        print(e)
        return jsonify({'Error': 'Invalid request'}), 500

    req = {'input': args}
    requests_queue.put(req)

    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)

    return jsonify(req['output'])


@app.route('/healthz', methods=["GET"])
def health_check():
    return "Health", 200


@app.route('/')
def main():
    # return render_template('main.html'), 200
    return 'hello', 200


if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0')