# AI-Role-assignment
1. Installation

To install the required libraries, run the following command:

```bash
python -m pip install -r requirements.txt
```
Training the Model
If you want to train the model, execute the following command:

```bash
python model.py
```
Pre-trained weights are also attached if you don't want to train the model.

2. Running the FastAPI Application
To run the FastAPI application and store input data in MySQL, follow these steps:

      1. Start MySQL and create the necessary schema.
      2. Set the database URL in the main.py script.
      3. Run the FastAPI application:
```bash

python main.py
```
3. Containerizing the AI Model and Web Service using Docker
Build the Docker image with the following command:

```bash

docker build -t classifiermodel #customize image name
```
Run the Docker container:

```bash

docker run -p 8000:80 classifiermodel #customize image name
```
The Docker image will be created, and the container will run.

4. Deploying with Kubernetes
Apply the Kubernetes deployment config:
```bash

kubectl apply -f modeldeploy.yaml
```
Apply the Kubernetes service config:
```bash

kubectl apply -f modeldeploy-service.yaml
```
Run the following command to deploy the Docker container:
```bash

kubectl apply -f modeldeploy.yaml
```
# NOTE 

YOU CAN DOWNLOAD THE DATASET FROM THE FOLLOWING LINK:- https://www.kaggle.com/datasets/tongpython/cat-and-dog/download?datasetVersionNumber=1
