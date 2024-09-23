# Use the official Python image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy current directory contents into the container
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 to the outside world
EXPOSE 8080

# Run the app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
