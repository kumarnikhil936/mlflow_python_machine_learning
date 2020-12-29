# MLflow Examples

---

This is just a subset of what has beed provided here: https://github.com/amesar/mlflow-examples

--- 


MLflow examples in Python for sklearn and Tensorflow / Keras machine learning models.

## Examples

* [sklearn](sklearn_wine) - Scikit-learn model - train and score. 
  * Canonical example that shows multiple ways to train and score.
  * Train locally or against a Databricks cluster.
  * Score real-time against a local web server or Docker container.
* [Keras/Tensorflow](keras_mnist) - train and score. ONNX working too.
  * Keras with TensorFlow 2.x 
  * [keras_tf_mnist](python/keras_tf_mnist) - MNIST dataset
  


## Setup

Use Python 3.7.5

* For Python environment use either:
  * Miniconda with [conda.yaml](python/conda.yaml).
  * Virtual environment with PyPi.

### Miniconda

* Install miniconda3: ``https://conda.io/miniconda.html``
* Create the environment: ``conda env create --file conda.yaml``
* Source the environment: `` source activate mlflow_venv``

### Virtual Environment

Create a virtual environment.
```
python -m venv mlflow_venv
source mlflow_venv/bin/activate
```

`pip install` the libraries in conda.yaml.

## MLflow Server

You can either run the MLflow tracking server directly on your laptop or with Docker.

#### Docker 

See [docker/docker-server/README](docker/docker-server/README.md).

#### Laptop Tracking Server

You can either use the local file store or a database-backed store. 
See MLflow [Storage](https://mlflow.org/docs/latest/tracking.html#storage) documentation.

The new MLflow 1.4.0 Model Registry functionality seems only to work with the database-backed store.


### Database Store

#### File Store

Start the MLflow tracking server.

```
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri $PWD/mlruns --default-artifact-root $PWD/mlruns
```

#### Database-backed store - MySQL

* Install MySQL
* Create an mlflow user with password.
* Create a database `mlflow` 

Start the MLflow Tracking Server
```
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri mysql://MLFLOW_USER:MLFLOW_PASSWORD@localhost:3306/mlflow \
  --default-artifact-root $PWD/mlruns  
```

#### Database-backed store - SQLite

```
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root $PWD/mlruns  
```

### Setup
Before running an experiment
```
export MLFLOW_TRACKING_URI=http://localhost:5000
```


### Data
Data for either of the two types of models are inside the individual folders.

