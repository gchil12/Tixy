from flask import Flask, request, jsonify
import openai
import pinecone
import requests
from google.cloud import secretmanager
import logging

# Initialize Flask app
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Starting Flask App")

# Access secrets from Google Cloud Secret Manager
def access_secret_version(secret_id):
    try:
        logger.info(f"Accessing secret: {secret_id}")
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/tickets-435609/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        logger.error(f"Error accessing secret {secret_id}: {e}")
        raise

# Load API keys from Google Cloud Secret Manager
try:
    OPENAI_API_KEY = access_secret_version("OPENAI_ORGANIZER_EVENTS_EMBEDDINGS")
    PINECONE_API_KEY = access_secret_version("PINECONE_API_KEY")
    PINECONE_ENV = access_secret_version("PINECONE_ENV")
    MANYCHAT_API_TOKEN = access_secret_version("MANYCHAT_API_TOKEN")

    logger.info(f"Successfully accessed API keys: OpenAI Key - {OPENAI_API_KEY[:4]}***, Pinecone Key - {PINECONE_API_KEY[:4]}***")
except Exception as e:
    logger.error(f"Failed to load API keys: {e}")
    raise

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone for organizer and event indexes using the new API
try:
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    logger.info("Pinecone initialized successfully")

    # Check if the index already exists before creating it
    if "tixy-organizers" not in pc.list_indexes().names():
        pc.create_index(
            name="tixy-organizers", 
            dimension=1536, 
            metric="euclidean", 
            spec=pinecone.ServerlessSpec(cloud='aws', region='us-west-2')
        )
        logger.info("Created 'tixy-organizers' index")
    else:
        logger.info("'tixy-organizers' index already exists")

    if "tixy-events" not in pc.list_indexes().names():
        pc.create_index(
            name="tixy-events", 
            dimension=1536, 
            metric="euclidean", 
            spec=pinecone.ServerlessSpec(cloud='aws', region='us-west-2')
        )
        logger.info("Created 'tixy-events' index")
    else:
        logger.info("'tixy-events' index already exists")

    organizer_index = pc.Index("tixy-organizers")
    event_index = pc.Index("tixy-events")
    logger.info("Pinecone indexes retrieved successfully")

except Exception as e:
    logger.error(f"Error initializing Pinecone: {e}")
    raise

# Send message to ManyChat function
def send_to_manychat(messenger_user_id, content_id, error_message=None):
    try:
        logger.info(f"Sending message to ManyChat for user {messenger_user_id} with content ID {content_id}")
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
        if response.status_code == 200:
            logger.info("Message sent successfully to ManyChat")
        else:
            logger.error(f"Failed to send message to ManyChat: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error sending message to ManyChat: {e}")
        return False

# /register-organizer function
@app.route('/register-organizer', methods=['POST'])
def register_organizer():
    try:
        data = request.json
        messenger_user_id = data['messenger_user_id']
        organizer_name = data['organizer_name']
        organizer_email = data['organizer_email']

        # Check if organizer email already exists in Pinecone
        existing_organizers = organizer_index.fetch(ids=[organizer_email])

        # If an organizer with the given email exists
        if existing_organizers['vectors']:
            send_to_manychat(messenger_user_id, "content20240917152105_198581", "Organizer with this email already exists")
            logger.warning(f"Organizer with email {organizer_email} already exists")
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
        logger.info(f"Organizer {organizer_name} registered successfully")

        # Notify success through ManyChat
        send_to_manychat(messenger_user_id, "content20240917151147_157784")
        return jsonify({"message": "Organizer registered successfully"}), 200
    except Exception as e:
        logger.error(f"Error in /register-organizer: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
