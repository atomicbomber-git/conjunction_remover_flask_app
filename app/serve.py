from flask import Flask, request, jsonify
import os
import nltk
from nltk.tag import CRFTagger
import numpy as np

app = Flask(__name__)

ct = CRFTagger()
ct.set_model_file(os.path.dirname(os.path.abspath(__file__)) + '/all_indo_man_tag_corpus_model.crf.tagger')

@app.route('/', methods=['POST'])
def process():
    # Tokenize input text
    input_text = nltk.word_tokenize(request.form.get('input', ''))

    # Tag sentence
    result = ct.tag_sents([input_text])

    # Remove unwanted elements
    forbidden_tags = ['SC', 'IN', 'CC']
    for index, sentence in enumerate(result):
        result[index] = [word for word in sentence if word[1] not in forbidden_tags]

    # Assemble output
    output = ''
    for sentence in result:
        for word in sentence:
            output = output + ' ' + word[0]

    return jsonify({
        'result': output.strip(),
        'status': 'success'
    })

if __name__ == '__main__':
    app.run()