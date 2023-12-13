# AI-Role-assignment
Installation
To install the required libraries, run the following command:

bash
Copy code
python -m pip install -r requirements.txt
Training the Model
If you want to train the model, execute the following command:

bash
Copy code
python model.py
Pre-trained weights are also attached if you don't want to train the model.

Running the FastAPI Application
To run the FastAPI application and store input data in MySQL, follow these steps:

Start MySQL and create the necessary schema.
Set the database URL in the main.py script.
Run the FastAPI application:
bash
Copy code
python main.py
Containerizing the AI Model and Web Service using Docker
Build the Docker image with the following command:

bash
Copy code
docker build -t classifiermodel #customize image name
Run the Docker container:

bash
Copy code
docker run -p 8000:80 classifiermodel #customize image name
The Docker image will be created, and the container will run.

Deploying with Kubernetes
Apply the Kubernetes deployment config:
bash
Copy code
kubectl apply -f modeldeploy.yaml
Apply the Kubernetes service config:
bash
Copy code
kubectl apply -f modeldeploy-service.yaml
Run the following command to deploy the Docker container:
bash
Copy code
kubectl apply -f modeldeploy.yaml
