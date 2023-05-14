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
stop_token = '\n'
for i in test_data:
    text = 'Given an original story and an intervening counterfactual event, the task is to minimally revise the story to make it compatible with the given counterfactual event.\n\n'
    text += 'premise: ' + i['premise'] + '\ninitial event: ' + i['initial'] + '\noriginal ending: ' + i['original_ending']
    text += '\ncounterfactual event: ' + i['counterfactual'] + '\nnew ending:'

    prompt[i['story_id']] = text
    
@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.ServiceUnavailableError))
def completions_with_backoff(prompt):
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=[stop_token]
    )
    return response


prediction_filename = 'davinci002_counterfactual_output'
prediction = {}

for i in test_data:
    logging.info(prompt[i['story_id']])
    response = completions_with_backoff(prompt[i['story_id']])['choices'][0]['text'].strip()
    logging.info(response)
    prediction[i['story_id']] = response

with open(prediction_filename+'.json', 'w') as f:
    f.write(json.dumps(prediction, indent=2))