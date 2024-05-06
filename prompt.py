import pandas as pd
from sklearn.model_selection import train_test_split
import ast
from dotenv import load_dotenv
from openai import OpenAI


if __name__ == '__main__':
    load_dotenv()
    client = OpenAI()

    # preprocess dev data
    dataset = pd.read_csv('data/annotations.csv', usecols=['doc_id', 'original_text', 'all'])
    dataset = dataset.sort_values(by='doc_id')
    train, test_dev = train_test_split(dataset, test_size=0.2, random_state=24)
    dev, test = train_test_split(test_dev, test_size=0.5, random_state=24)
    test['all'] = test['all'].map(ast.literal_eval)

    with open('prompt_docs/prompt1.txt', encoding='utf-8') as f:
        prompt_part1 = f.read() + ' '
    prompt_part3 = """ ”
What is the answer?"""

    for i, row in enumerate(test.itertuples()):
        doc_id = row.doc_id
        paragraph = row.original_text
        prompt = prompt_part1 + paragraph + prompt_part3
        print(prompt)

        # GPT request
        messages = [{"role": "user", "content": prompt}]
        completion = client.chat.completions.create(
          model="gpt-3.5-turbo-0125",
          messages=messages
        )
        response = completion.choices[0].message.content
        print(response)

        # save output to a file
        with open('prompt_docs/gpt_responses1_test.txt', 'a') as f:
            f.write('ID: ' + str(doc_id) + '\n' + response + '\n')
