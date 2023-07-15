# a flask server that calls the spacy model and returns the results

from flask import Flask, request, jsonify, render_template
import spacy
import flask_cors


app = Flask(__name__, static_url_path='/static')
flask_cors.CORS(app, origins="*")

# load the spacy model
nlp = spacy.load("model")

def predict_labels(text) -> list:
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def get_the_sentences(entities: list, text: str) -> dict:
    res = {}
    for entity in entities:
        # find the last fullstop before the start of the entity
        entity_start = text.find(entity[0])
        entity_end = entity_start + len(entity[0])
        start = text.rfind(".", 0, entity_start) + 1
        # find the next fullstop after the end of the entity
        end = text.find(".", entity_end) + 1
        # print the sentence containing the entity
        res[text[start:end].strip()] = entity[1]
    return res

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# create a route for the server
@app.route('/api', methods=['POST'])
def api():
    data = request.get_json(force=True)
    text = data['text']
    entities = predict_labels(text)
    sentences = get_the_sentences(entities, text)
    for sentence, label in sentences.items():
        text = text.replace(sentence, f"<{label}>{sentence}</{label}>")
    return jsonify({'text': text})
    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)
