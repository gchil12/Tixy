from flask import Flask, request, jsonify
import openai
import pinecone
import requests
from google.cloud import secretmanager

# Initialize Flask app
app = Flask(__name__)

# Access secrets from Google Cloud Secret Manager
def access_secret_version(secret_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/tickets-435609/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Load API keys from Google Cloud Secret Manager
OPENAI_API_KEY = access_secret_version("OPENAI_ORGANIZER_EVENTS_EMBEDDINGS")
PINECONE_API_KEY = access_secret_version("PINECONE_API_KEY")
PINECONE_ENV = access_secret_version("PINECONE_ENV")
MANYCHAT_API_TOKEN = access_secret_version("MANYCHAT_API_TOKEN")

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone for organizer and event indexes using the new API
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Check if the index already exists before creating it
if "tixy-organizers" not in pc.list_indexes().names():
    pc.create_index(
        name="tixy-organizers", 
        dimension=1536, 
        metric="euclidean", 
        spec=pinecone.ServerlessSpec(cloud='aws', region='us-west-2')
    )

if "tixy-events" not in pc.list_indexes().names():
    pc.create_index(
        name="tixy-events", 
        dimension=1536, 
        metric="euclidean", 
        spec=pinecone.ServerlessSpec(cloud='aws', region='us-west-2')
    )

# Use the correct method to retrieve indexes
organizer_index = pc.Index("tixy-organizers")
event_index = pc.Index("tixy-events")

# Send message to ManyChat function
def send_to_manychat(messenger_user_id, content_id, error_message=None):
    url = "https://api.manychat.com/fb/sending/sendContent"
    headers = {
        "Authorization": f"Bearer {MANYCHAT_API_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "subscriber_id": messenger_user_id,
        "content_id": content_id
    }
    if error_message:
        data["error_message"] = error_message
    response = requests.post(url, headers=headers, json=data)
    return response.status_code == 200

# /register-organizer function
@app.route('/register-organizer', methods=['POST'])
def register_organizer():
    data = request.json
    messenger_user_id = data['messenger_user_id']
    organizer_name = data['organizer_name']
    organizer_email = data['organizer_email']

    # Check if organizer email already exists in Pinecone
    existing_organizers = organizer_index.fetch(ids=[organizer_email])

    # If an organizer with the given email exists
    if existing_organizers['vectors']:
        send_to_manychat(messenger_user_id, "content20240917152105_198581", "Organizer with this email already exists")
        return jsonify({"error": "Organizer with this email already exists"}), 409

    # Create an embedding for organizer name using the new OpenAI API
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=[organizer_name]
    )
    
    # Access the embedding correctly
    embedding = response.data[0].embedding

    # Upsert organizer data into Pinecone with organizer_email as the ID
    organizer_vector = (organizer_email, embedding, {
        'organizer_name': organizer_name,
        'organizer_email': organizer_email,
        'messenger_user_id': messenger_user_id,
        **data
    })
    organizer_index.upsert([organizer_vector])

    # Notify success through ManyChat
    send_to_manychat(messenger_user_id, "content20240917151147_157784")
    return jsonify({"message": "Organizer registered successfully"}), 200
