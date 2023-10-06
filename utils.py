import nltk
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from jira import JIRA
import string


nltk.download('punkt')

# Preprocess text data
def preprocess_text(text):
    # If the text is already a list (from previous preprocessing), just return it
    if isinstance(text, list):
        return text

    # Handle None type or other non-string types
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    return tokens

# Fetch data from JIRA based on a ticket number
def fetch_from_jira(ticket_number, jira_instance):
    issue = jira_instance.issue(ticket_number)
    summary = issue.fields.summary
    description = issue.fields.description or ''
    reporter = issue.fields.reporter.displayName
    affected_release = issue.fields.versions
    if affected_release:
        affected_release = ','.join([version.name for version in affected_release])
    else:
        affected_release = ''
    team = issue.fields.customfield_16700
    if team:
        team_value = team.name
    else:
        team_value = 'Unknown'
    return {
        "Summary": summary,
        "Description": description,
        "Reporter": reporter,
        "Affected Release": affected_release,
        "Team": team_value
    }

def predict_team_from_jira(ticket_number, model, jira_instance, vectorizer, encoder_reporter, encoder_release, encoder_team):
    # Fetch the JIRA ticket data
    issue_data = fetch_from_jira(ticket_number, jira_instance)
    
    # Create a temporary dataframe to hold this data
    temp_df = pd.DataFrame([issue_data])

    # Preprocess this data
    temp_df['Summary'] = temp_df['Summary'].apply(preprocess_text)
    temp_df['Description'] = temp_df['Description'].apply(preprocess_text)
    temp_df['Text'] = temp_df['Summary'] + temp_df['Description']
    temp_df['Text'] = temp_df['Text'].apply(lambda x: ' '.join(x))
    X_text_features = vectorizer.transform(temp_df['Text'])
    temp_df['Reporter_encoded'] = encoder_reporter.transform(temp_df['Reporter'])
    temp_df['Affected Release'] = temp_df['Affected Release'].apply(lambda x: ','.join(map(str, x.split(','))) if x else '')
    affected_release_encoded = encoder_release.transform(temp_df[['Affected Release']])
    
    # Combine the features
    X_new = np.hstack((X_text_features.toarray(), temp_df['Reporter_encoded'].values.reshape(-1,1), affected_release_encoded))

    # Predict using the model
    predicted_team_encoded = model.predict(X_new)

    # Decode the predicted team
    predicted_team = encoder_team.inverse_transform(predicted_team_encoded)

    return predicted_team[0]
