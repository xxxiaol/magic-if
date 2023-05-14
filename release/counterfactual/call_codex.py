import json
import re
import logging
import backoff

logging.basicConfig(level=logging.INFO)

import os
import openai

os.environ["OPENAI_API_KEY"] = ""  # paste the openai api key
openai.api_key = os.getenv("OPENAI_API_KEY")

source_file = 'test_data_new.json'
test_data = []
with open(source_file, 'r') as f:
    for line in f:
        test_data.append(json.loads(line))

prompt = {}
stop_token = '# end'
for i in test_data:
    tmp = re.split('(\.|!|\?)', i['original_ending'])
    assert len(tmp) == 7, (idxi, i['original_ending'])
    original_ending = []

    for j in range(3):
        original_ending.append((tmp[j*2]+tmp[j*2+1]).strip())
        
    text = '# task: generate an ending with three sentences given the premise and the hypothesis\n\n'
    text += 'def main():\n    premise()\n    if hypothesis_1():\n        ending_1()\n    elif hypothesis_2():\n        # minimally revise ending_1\n        ending_2()  \n\ndef premise():\n    # '
    text += i['premise'] + '\n    \ndef hypothesis_1():\n    # ' + i['initial'] + '\n    \ndef hypothesis_2():\n    # '
    text += i['counterfactual'] + '\n    \ndef ending_1():\n'
    for j in range(3):
        text += '    # ' + original_ending[j] + '\n'
    text += '    # end\n    \ndef ending_2():\n'
    
    text = text.replace('\t', '    ')
    prompt[i['story_id']] = text

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


prediction_filename = 'codex_counterfactual_output'
prediction = {}

for i in test_data:
    logging.info(prompt[i['story_id']])
    response = completions_with_backoff(prompt[i['story_id']])['choices'][0]['text'].strip()
    logging.info(response)
    prediction[i['story_id']] = response

with open(prediction_filename+'.json', 'w') as f:
    f.write(json.dumps(prediction, indent=2))