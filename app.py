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

@app.route('/')
def home():
    return "<h1>Welcome to Tixy App</h1><p>This is the home page.</p>"

# Function to validate event data using OpenAI
def validate_event_data(event_location, event_start_date, event_end_date, event_location_map, event_graphics):
    prompt = f"""Check the validity of the following event data:
    1. Event location: "{event_location}" - should be precise enough for visitors to navigate.
    2. Event start date: "{event_start_date}" - should be earlier than event end date.
    3. Event end date: "{event_end_date}".
    4. Event location map: "{event_location_map}" - should be a valid Google Maps link.
    5. Event graphics: "{event_graphics}" - should be a valid image link.

    Return a JSON object where each field has either 'OK' or an error message if invalid."""
    
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=300
        )
        validation_result = response.choices[0].text.strip()
        return validation_result
    except Exception as e:
        logger.error(f"Error in OpenAI API call for validation: {e}")
        return None

@app.route('/create-event', methods=['POST'])
def create_event():
    try:
        # Extract data from the request
        data = request.json
        organizer_email = data.get('organizer_email')
        event_title = data.get('event_title')
        event_description = data.get('event_description')
        event_start_date = data.get('event_start_date')
        event_end_date = data.get('event_end_date')
        event_location = data.get('event_location')
        event_location_map = data.get('event_location_map')
        event_graphics = data.get('event_graphics')

        # Validate required fields
        if not all([organizer_email, event_title, event_start_date, event_end_date, event_location]):
            return jsonify({"error": "All fields are required"}), 400

        # Validate event data using OpenAI
        validation_result = validate_event_data(event_location, event_start_date, event_end_date, event_location_map, event_graphics)
        
        if validation_result is None:
            return jsonify({"error": "Failed to validate event data"}), 500

        validation_result_json = eval(validation_result)  # convert the OpenAI response to JSON

        # Check for errors in validation result
        if any(v != 'OK' for v in validation_result_json.values()):
            return jsonify({"error": "Event data validation failed", "details": validation_result_json}), 400

        # Fetch the organizer's data from Pinecone using their email
        existing_organizer = organizer_index.fetch(ids=[organizer_email])
        if not existing_organizer['vectors']:
            return jsonify({"error": "Organizer not found"}), 404

        # Create an embedding for the event description using OpenAI
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=[event_description]
        )
        event_embedding = response.data[0].embedding

        # Upsert event data into Pinecone with a unique event ID
        event_id = f"{organizer_email}-{event_title}-{event_start_date}"
        event_vector = (event_id, event_embedding, {
            'organizer_email': organizer_email,
            'event_title': event_title,
            'event_description': event_description,
            'event_start_date': event_start_date,
            'event_end_date': event_end_date,
            'event_location': event_location,
            'event_location_map': event_location_map,
            'event_graphics': event_graphics
        })
        event_index.upsert([event_vector])
        logger.info(f"Event {event_title} added successfully for {organizer_email}")

        # Update ManyChat event_addition attribute to 'success'
        if update_manychat_user_attribute(data.get('messenger_user_id'), "event_addition", "success"):
            logger.info(f"Event addition status updated for user {data.get('messenger_user_id')}")
        else:
            logger.error(f"Failed to update event_addition status for user {data.get('messenger_user_id')}")

        return jsonify({"message": f"Event {event_title} created successfully"}), 200
    except Exception as e:
        logger.error(f"Error in /create-event: {e}")
        return jsonify({"error": str(e)}), 500

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

# Initialize Pinecone for organizer and event indexes
try:
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    logger.info("Pinecone initialized successfully")

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

# Function to update ManyChat user attribute
def update_manychat_user_attribute(messenger_user_id, field_name, field_value):
    url = "https://api.manychat.com/fb/subscriber/setCustomFieldByName"
    headers = {
        "Authorization": f"Bearer {MANYCHAT_API_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "subscriber_id": messenger_user_id,
        "field_name": field_name,
        "field_value": field_value
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response_data = response.json()
        logger.info(f"Update Attribute Response: Status Code: {response.status_code}, Response: {response_data}")
        
        if response.status_code == 200 and response_data.get('status') == 'success':
            return True
        else:
            return False
    except Exception as e:
        logger.error(f"Error updating user attribute in ManyChat: {e}")
        return False

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
