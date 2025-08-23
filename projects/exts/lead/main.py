import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, select, insert

metadata = MetaData()
response_table = Table(
  "response",
  metadata,
  Column("email", String, primary_key=True),
  Column("name", String),
  Column("number", Integer)
)

# Create an engine (adjust connection string as needed)
engine = create_engine('sqlite:///lead.db')

def get_response_by_email(email: str) -> None:
  stmt = select(response_table).where(response_table.c.email == email)
  with engine.connect() as conn:
    result = conn.execute(stmt)
    for row in result:
      print(row)

def extract_answers(data, field_mapping=None):
  """Extract all answers from the JSON response with optional field mapping"""
  answers = {}
  
  # Default mapping if none provided
  if field_mapping is None:
    field_mapping = {
      '7aef9547': 'name',
      '71ef93f7': 'number', 
      '61d54db6': 'email'
    }
  
  # Navigate through the JSON structure
  for response in data['responses']:
    for question_id, question_data in response['answers'].items():
      # Extract the first answer value (assuming single answers)
      answer_value = question_data['textAnswers']['answers'][0]['value']
      
      # Use mapped name if available, otherwise use original question_id
      field_name = field_mapping.get(question_id, question_id)
      print(field_name, answer_value)
      answers[field_name] = answer_value
  
  return answers

def save_responses_to_db(responses_data, field_mapping=None):
  """Extract responses and save them to the database"""
  # Create tables if they don't exist
  metadata.create_all(engine)
  
  extracted_answers = extract_answers(responses_data, field_mapping)
  
  with engine.connect() as conn:
    # Convert number to int if it exists
    if 'number' in extracted_answers:
      try:
        extracted_answers['number'] = int(extracted_answers['number'])
      except ValueError:
        extracted_answers['number'] = 0
    
    # Check if email already exists
    email = extracted_answers.get('email')
    if email:
      existing = conn.execute(
        select(response_table).where(response_table.c.email == email)
      ).fetchone()
      
      if existing:
        # Update existing record
        stmt = response_table.update().where(
          response_table.c.email == email
        ).values(**extracted_answers)
        conn.execute(stmt)
        print(f"Updated response for email: {email}")
      else:
        # Insert new record
        stmt = insert(response_table).values(**extracted_answers)
        conn.execute(stmt)
        print(f"Inserted new response for email: {email}")
    else:
      # Insert without email check
      stmt = insert(response_table).values(**extracted_answers)
      conn.execute(stmt)
      print("Inserted response without email")
    
    conn.commit()

# The scope determines the level of access.
# This scope is read-only for form responses.
SCOPES = ["https://www.googleapis.com/auth/forms.responses.readonly"]

# The ID of the Google Form.
FORM_ID = input("Form ID:") 

def main():
  """
  Authenticates with the Google Forms API and prints responses from a form.
  """
  creds = None
  # The file token.json stores the user's access and refresh tokens, and is
  # created automatically when the authorization flow completes for the first
  # time.
  if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
  # If there are no (valid) credentials available, let the user log in.
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      # IMPORTANT: Rename your downloaded client_secrets.json to credentials.json
      flow = InstalledAppFlow.from_client_secrets_file(
        "credentials.json", SCOPES
      )
      creds = flow.run_local_server(port=0)
      
    # Save the credentials for the next run
    with open("token.json", "w") as token:
      token.write(creds.to_json())
  try:
    # Build the service object
    service = build("forms", "v1", credentials=creds)
    # Retrieve the form responses.
    result = service.forms().responses().list(formId=FORM_ID).execute()
    print("Raw API Response:")
    print(result)
    
    # Process and save responses to database
    if 'responses' in result:
      # You may need to adjust the field mapping based on your actual question IDs
      field_mapping = {
        # Replace these with your actual question IDs from the form
        '7aef9547': 'name',
        '71ef93f7': 'number',
        '61d54db6': 'email'
      }
      
      save_responses_to_db(result, field_mapping)
      
      # Example: Query a specific email
      email_to_search = input("Enter email to search (or press Enter to skip): ")
      if email_to_search:
        get_response_by_email(email_to_search)
    else:
      print("No responses found in the form.")
      
  except HttpError as err:
    print(f"An error occurred: {err}")

if __name__ == "__main__":
  main()
