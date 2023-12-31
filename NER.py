import sqlite3
import pandas as pd
# import spacy
# import sys

pd.set_option("display.max_colwidth", 60)
pd.set_option("display.max_rows", 100)

RELEASE_DB_NAME = "release_db.sqlite"
conn = sqlite3.connect(RELEASE_DB_NAME)

policy_texts_df = pd.read_sql_query("SELECT * FROM policy_texts", conn)
simply_the_text = policy_texts_df[["id", "policy_text"]]
simply_the_text.head()
# choose a subset of the policies to label (the more, the longer it takes to generate the model)
simply_the_text = simply_the_text[:500]
# use labels.json to label policies in simply_the_text dataframe
import json
with open('labels.json') as f:
    labels = json.load(f)
# create a new column in simply_the_text dataframe called "label"
simply_the_text["label"] = ""

# for each policy in simply_the_text dataframe, find the corresponding label in labels.json
for i in range(len(simply_the_text)):
    for title, labels_list in labels.items():
        # if any word in labels_list appears in policy_text, then add the title in the labels. there can be multiple labels for each policy
        if any(word in simply_the_text["policy_text"][i] for word in labels_list):
            simply_the_text["label"][i] +=  ", " + title if simply_the_text["label"][i] != "" else title

import json
import pandas as pd

with open('labels.json') as f:
    labels = json.load(f)

# Create a new column in `simply_the_text` dataframe called "annotations"
simply_the_text["annotations"] = ""

# Iterate over each policy in `simply_the_text` dataframe
for i, row in simply_the_text.iterrows():
    policy_text = row["policy_text"]
    annotations = []
    seen_tokens = set()  # Track the tokens that have been seen
    
    # Iterate over each label in labels.json
    for label, label_keywords in labels.items():
        # Check if any word in `label_keywords` appears in `policy_text`
        for keyword in label_keywords:
            start = policy_text.find(keyword)
            while start != -1:
                end = start + len(keyword)
                if all(token not in seen_tokens for token in range(start, end)):
                    annotations.append((start, end, label))
                    seen_tokens.update(range(start, end))
                start = policy_text.find(keyword, end)
    
    # Add the annotations to the "annotations" column
    simply_the_text.at[i, "annotations"] = {"entities": annotations}

import pandas as pd
import spacy
import random
from spacy.util import minibatch, compounding
from spacy.training.example import Example

train_data = simply_the_text.sample(frac=0.8, random_state=42)
test_data = simply_the_text.drop(train_data.index)
nlp = spacy.blank("en")  # Use an appropriate spaCy model

# Prepare the training data in the required format for NER
train_data_ner = []
for _, row in train_data.iterrows():
    doc = nlp.make_doc(row["policy_text"])
    annotations = row["annotations"]
    
    example = Example.from_dict(doc, annotations)
    train_data_ner.append(example)

# Train the NER model
print("Training the NER model...")

# Add the NER component to the pipeline
if "ner" not in nlp.pipe_names:
    ner = nlp.create_pipe("ner")
    nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# get all label titles from labels.json
import json 
with open('labels.json') as f:
    labels = json.load(f)
labels = list(labels.keys())
print(labels)

for label in labels:
    ner.add_label(label)

# Disable other pipeline components for training efficiency
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    batch_sizes = compounding(4.0, 32.0, 1.001)
    epochs = 8
    all_losses = []
    for epoch in range(epochs):
        random.shuffle(train_data_ner)
        losses = {}
        batches = minibatch(train_data_ner, size=batch_sizes)
        print(f"Epoch {epoch+1}/{epochs}")
        for batch in batches:
            nlp.update(batch, drop=0.3, losses=losses)
        all_losses.append(losses)
        print(f"Loss: {losses['ner']:.4f}")

        # if loss increases, stop training
        if epoch > 0 and losses["ner"] > all_losses[epoch-1]["ner"]:
            break


# Evaluate the model on the testing set
print("Evaluating the model on the testing set...")
test_data_ner = []
for _, row in test_data.iterrows():
    doc = nlp.make_doc(row["policy_text"])
    annotations = row["annotations"]
    example = Example.from_dict(doc, annotations)
    test_data_ner.append(example)

# Make predictions using the trained model
print("Making predictions using the trained model...")
predictions = []
for example in test_data_ner:
    doc = nlp(example.text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    predictions.append(entities)

# Evaluate the predictions
print("Evaluating the predictions...")
results = pd.DataFrame(columns=["text", "label", "prediction"])
for (_, row), prediction in zip(test_data.iterrows(), predictions):
    text = row["policy_text"]
    label = row["label"]
    prediction = ", ".join([f"{ent[0]} ({ent[1]})" for ent in prediction])
    results = results.add(pd.Series([text, label, prediction], index=results.columns), ignore_index=True)

# Display the evaluation results

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# Plot the confusion matrix
plt.figure(figsize=(20, 20))
sns.heatmap(pd.crosstab(results["label"], results["prediction"], rownames=["Actual"], colnames=["Predicted"]))
plt.show()

# Print the classification report
print(classification_report(results["label"], results["prediction"]))

# Save the trained model
nlp.to_disk("model")

nlp = spacy.load("trained_ner_model")

# Define a function to predict labels for a given text
def predict_labels(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Define a function to display the predicted labels
def display_predictions(text):
    entities = predict_labels(text)
    for entity in entities:
        # find the last fullstop before the start of the entity
        entity_start = text.find(entity[0])
        entity_end = entity_start + len(entity[0])
        start = text.rfind(".", 0, entity_start) + 1
        # find the next fullstop after the end of the entity
        end = text.find(".", entity_end) + 1
        # print the sentence containing the entity
        print(f"{text[start:end].strip()} : {entity[1]}")
        

# Predict labels for the given text
# get the 501th policy text
text = policy_texts_df[["id", "policy_text"]]
print(text.iloc[501, 1])
display_predictions(text.iloc[501, 1])
