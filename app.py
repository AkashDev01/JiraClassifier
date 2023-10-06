from flask import Flask, request, jsonify
from utils import predict_team_from_jira, preprocess_text, fetch_from_jira
from jira import JIRA
import getpass
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import nltk
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from jira import JIRA
import getpass
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["https://www.postman.co"])


# Initialize JIRA connection
host = "http://jira.trimble.tools"
uid = "akash_v@trimble.com"
# pswd = getpass.getpass('Password:')
pswd = "VRjrhTuFp000rEJglt74ADlELgKs6VydFRBmeN"
jira = JIRA(server=host, basic_auth=(uid, pswd))

# Fetch all projects
projects = jira.projects()

# Initialize lists
project_grid = list()
project_list = list()

# Loop through each project and append its key to project_list
for project in projects:
    project_list.append(project.key)

# Print the list of project keys
# print(project_list)

# Fetch issues for a specific project 
# issues = jira.search_issues("project=AGAM")
def fetch_all_issues(project_key):
    issues_list = []
    start_at = 0
    max_results = 2000 
    i=0
    # while True:
    issues = jira.search_issues(
        'project={}'.format(project_key),
        startAt=start_at,
        maxResults=max_results
    )
        # if len(issues) == 0 or i >= 20:
        #     break  # Break out of the loop if no more issues are returned
    issues_list.extend(issues)
    start_at += len(issues)  # Update the starting point for the next batch of issues

    return issues_list


project_key = 'AGAM'
issues = fetch_all_issues(project_key)
for issue in issues:
    print(issue.key, issue.fields.summary)
# Initialize an empty list to store issue data
issue_data = []

# Loop through each issue and extract necessary fields
for issue in issues:
    summary = issue.fields.summary
    description = issue.fields.description
    reporter = issue.fields.reporter.displayName
    affected_release = issue.fields.versions
    # team = issue.fields.customfield_16700.name
    team = issue.fields.customfield_16700
    if team is not None:
        team_value = team.name
    else:
        team_value = 'Unknown'
    issue_data.append([summary, description, reporter, affected_release, team_value])

# Convert the list to a Pandas DataFrame
df = pd.DataFrame(issue_data, columns=['Summary', 'Description', 'Reporter', 'Affected Release', 'Team'])
 
print(df.count())

import nltk
nltk.download('punkt')

import string
# Preprocess text
def preprocess_text(text):
    # Handle None type
    if not text:
        text = ''
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    return tokens

def preprocess_text(text):
    # If the text is already a list (from previous preprocessing), just return it
    import string
    if isinstance(text, list):
        return text

    # Handle None type or other non-string types
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    return tokens

# Preprocess the 'Summary' and 'Description' columns
df['Summary'] = df['Summary'].apply(preprocess_text)
df['Description'] = df['Description'].apply(preprocess_text)

# Combine Summary and Description
df['Text'] = df['Summary'] + df['Description']
df['Text'] = df['Text'].apply(lambda x: ' '.join(x))

# Vectorize the combined text
vectorizer = TfidfVectorizer(stop_words='english')
X_text_features = vectorizer.fit_transform(df['Text'])

# Encode the "Reporter" feature
encoder_reporter = LabelEncoder()
df['Reporter_encoded'] = encoder_reporter.fit_transform(df['Reporter'])

# One-hot encoding for "Affected Release"
# Convert lists to comma-separated strings
df['Affected Release'] = df['Affected Release'].apply(lambda x: ','.join(map(str, x)) if x else '')
encoder_release = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
affected_release_encoded = encoder_release.fit_transform(df[['Affected Release']])
affected_release_df = pd.DataFrame(affected_release_encoded, columns=encoder_release.get_feature_names_out(['Affected Release']))

# Encode target variable "Team"
encoder_team = LabelEncoder()
df['Team_encoded'] = encoder_team.fit_transform(df['Team'])

# Concatenate all the features
X = np.hstack((X_text_features.toarray(), df['Reporter_encoded'].values.reshape(-1,1), affected_release_df.values))
y = df['Team_encoded']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using RandomForest
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate the model
accuracy = clf.score(X_test, y_test)
print(f"Model Accuracy with RandomForest: {accuracy}")

# # Using Gradient Boosting
# from sklearn.ensemble import GradientBoostingClassifier

# clf_gb = GradientBoostingClassifier()
# clf_gb.fit(X_train, y_train)

# # Evaluate the Gradient Boosting model
# accuracy_gb = clf_gb.score(X_test, y_test)
# print(f"Model Accuracy with GradientBoosting: {accuracy_gb}")

@app.route('/')
def home():
    return "Hello, Flask!"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ticket_number = data['ticket_number']
    try:
        predicted_team = predict_team_from_jira(ticket_number, clf, jira, vectorizer, encoder_reporter, encoder_release, encoder_team)
        response = {
            'ticket_number': ticket_number,
            'predicted_team': predicted_team,
            'description': fetch_from_jira(ticket_number, jira)['Description'],
            'summary': fetch_from_jira(ticket_number, jira)['Summary']
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)