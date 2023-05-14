import json
import logging
import backoff

logging.basicConfig(level=logging.INFO)

import os
import openai

os.environ["OPENAI_API_KEY"] = ""  # paste the openai api key
openai.api_key = os.getenv("OPENAI_API_KEY")

source_file = 'test_cleanup_no_label.json'
with open(source_file, 'r') as f:
    sent_lst = json.load(f)

test_ref = 'test_cleanup_ref.json'
with open(test_ref, 'r') as f:
    test_ref_lst = json.load(f)
    
test_id = list(sent_lst.keys())

prompt = {}
stop_token = '\n'
for i in test_id:
    data = sent_lst[i]

    text = '# task: generate a plausible explanatory hypothesis given the premise and the ending\n'

    text += 'def main():\n\tpremise()\n\tif hypothesis():\n\t\tending()\n\ndef premise():\n\t# '
    text += data['obs1'] + '\n\ndef ending(): \n\t# ' + data['obs2'] + '\n\ndef hypothesis(): \n\t#'
    
    text = text.replace('\t', '    ')
    prompt[i] = text

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.ServiceUnavailableError))
def completions_with_backoff(prompt):
    response = openai.Completion.create(
        model="code-davinci-002",
        prompt=prompt,
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=[stop_token]
    )
    return response


prediction_filename = 'codex_abductive_output'
prediction = {}

for i in test_id:
    logging.info(prompt[i])
    response = completions_with_backoff(prompt[i])['choices'][0]['text'].strip()
    logging.info(response)
    prediction[i] = response

with open(prediction_filename+'.json', 'w') as f:
    f.write(json.dumps(prediction, indent=2))