# Firebase setup for VitaminCare

## 1. Firebase project
Open https://console.firebase.google.com/ and create a project.

Enable:
- Authentication → Sign-in method → Email/Password
- Firestore Database → Create database

## 2. Web API key
Firebase Console → Project settings → General → Your apps → Web app.
Copy the Web API key.

## 3. Service account
Firebase Console → Project settings → Service accounts → Generate new private key.
Download the JSON. NEVER upload it to GitHub.

## 4. Streamlit secrets
Create `.streamlit/secrets.toml` locally:

firebase_web_api_key = "YOUR_WEB_API_KEY"

[firebase_service_account]
type = "service_account"
project_id = "YOUR_PROJECT_ID"
private_key_id = "YOUR_PRIVATE_KEY_ID"
private_key = """-----BEGIN PRIVATE KEY-----
PASTE_PRIVATE_KEY_FROM_JSON
-----END PRIVATE KEY-----
"""
client_email = "YOUR_SERVICE_ACCOUNT_EMAIL"
client_id = "YOUR_CLIENT_ID"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "YOUR_CLIENT_CERT_URL"

Never commit `secrets.toml` or the service-account JSON.

## 5. Run
pip install -r requirements.txt
streamlit run app.py

## 6. Data structure
users/{uid}
  name
  email
  updatedAt

users/{uid}/assessments/{assessmentId}
  createdAt
  inputs
  results

Each user's reports are loaded using their Firebase UID.

## 7. Streamlit Cloud
Put the code and CSV on GitHub, then paste the same `secrets.toml` contents into the app's Secrets settings. Do not upload secrets to GitHub.
