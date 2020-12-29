# mlflow-examples - sklearn 

## Overview
* Wine Quality DecisionTreeRegressor example.
* Is a well-formed Python project that generates a wheel.
* This example demonstrates all features of MLflow training and prediction.
* Saves model in pickle format.
* Saves plot artifacts.
* Shows several ways to run training:
  * `mlflow run` - several variants. 
* Shows several ways to run prediction:
  * Real-time scoring
    * Local web server
    * Docker container - AWS SageMaker in local mode
  
* Data: [wine-quality-white.csv](wine-quality-white.csv)



## MLFlow Server

Run the mlflow server using to access the UI
```
mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 0.0.0.0
```


## Training

Source: [train.py](train.py).

There are several ways to train a model with MLflow.
  1. MLflow CLI `run` command
  1. Unmanaged without MLflow CLI

### Options

```
python train.py --help

Options:
  --experiment_name TEXT    Experiment name.
  --data_path TEXT          Data path.
  --model_name TEXT         Registered model name.
  --max_depth INTEGER       Max depth parameter.
  --max_leaf_nodes INTEGER  Max leaf nodes parameter.
  --output_path TEXT        Output file containing run ID.
  --log_as_onnx BOOLEAN     Log model as ONNX flavor. Default is false.
  --run_origin TEXT         Run origin.
  --autolog BOOLEAN         Autolog parameters and metrics. Default is False.
  --save_signature BOOLEAN  Save model signature. Default is False.
```

#### Autolog

If you set the `autolog` option to True, [mlflow.sklearn.autolog()](https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html#mlflow.sklearn.autolog) will be called and manually set parameters will not be recorded. Note that the model artifact path is simply model.

If you set the `autolog`  option, mannually set parameters will not be recorded.
Note that the model artifact path is simply `model`.

Here's the list of parameters for DecisionTreeRegressor:
* criterion       
* max_depth       
* max_features    
* max_leaf_nodes  
* min_impurity_decrease   
* min_impurity_split      
* min_samples_leaf        
* min_samples_split       
* min_weight_fraction_leaf        
* presort 
* random_state    
* splitter


### Running without MLflow CLI

Run the standard main function from the command-line.
```
python train.py --experiment_name sklearn --max_depth 2 --max_leaf_nodes 32
```

## Predictions

You can make predictions in two ways:
* Batch predictions - direct calls to retrieve the model and score large files.
  * mlflow.sklearn.load_model()
  * mlflow.pyfunc.load_model()
* Real-time predictions - use MLflow's scoring server to score individual requests.


### Batch Predictions

#### 1. Predict with mlflow.sklearn.load_model()

You can use either a `runs` or `models` scheme.

URI with `runs` scheme.
```
python predict.py mlruns\1\61d45b785b0742168e609bf5029e1479\artifacts\sklearn-model
```

Result.
```
predictions: [5.55109634 5.29772751 5.42757213 5.56288644 5.56288644]
```


Snippet from [predict.py](predict.py):
```
model = mlflow.sklearn.load_model(model_uri)
df = pd.read_csv("wine-quality-white.csv")
predictions = model.predict(data)
```


#### 2. Predict with mlflow.pyfunc.load_model()

```
python pyfunc_predict.py mlruns\1\61d45b785b0742168e609bf5029e1479\artifacts\sklearn-model
```

```
predictions: [5.55109634 5.29772751 5.42757213 5.56288644 5.56288644]
```
From [pyfunc_predict.py](pyfunc_predict.py):
```
data_path = "wine-quality-white.csv"
data = util.read_prediction_data(data_path)
model_uri = client.get_run(run_id).info.artifact_uri + "/sklearn-model"
model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(data)

```


### Real-time Predictions

Use a server to score predictions over HTTP.

There are several ways to launch the server:
  1. MLflow scoring web server 
  2. Plain docker container
  
See MLflow documentation:
* [Built-In Deployment Tools](https://mlflow.org/docs/latest/models.html#built-in-deployment-tools)
* [Tutorial - Serving the Model](https://www.mlflow.org/docs/latest/tutorial.html#serving-the-model)
* [Quickstart - Saving and Serving Models](https://www.mlflow.org/docs/latest/quickstart.html#saving-and-serving-models)

In one window launch the server.
```
mlflow models serve -m mlruns\1\61d45b785b0742168e609bf5029e1479\artifacts\sklearn-model -h 0.0.0.0 -p 5001
```

#### 1. MLflow scoring web server

Launch the web server.
```
mlflow models serve -m mlruns\1\61d45b785b0742168e609bf5029e1479\artifacts\sklearn-model -h 0.0.0.0 -p 5001
```

Make predictions with curl as described above.

#### 2. Plain Docker Container

See [build-docker](https://mlflow.org/docs/latest/cli.html#mlflow-models-build-docker) documentation.

First build the docker image.
```
mlflow models build-docker -m mlruns/1/61d45b785b0742168e609bf5029e1479/artifacts/sklearn-model -n dk_sklearn_wine
```

Then launch the server as a docker container.
```
docker run -p 5001:8080 dk_sklearn_wine
```
Make predictions with curl as described above.

In another window, score some data.
```
curl -X POST -H "Content-Type:application/json"  -d @./wine-quality.json http://localhost:5001/invocations
```
```
[
  [5.470588235294118,5.470588235294118,5.769607843137255]
]
```

Data should be in `JSON-serialized Pandas DataFrames split orientation` format
such as [wine-quality.json](wine-quality.json).
```
{
  "columns": [
    "alcohol",
    "chlorides",
    "citric acid",
    "density",
    "fixed acidity",
    "free sulfur dioxide",
    "pH",
    "residual sugar",
    "sulphates",
    "total sulfur dioxide",
    "volatile acidity"
  ],
  "data": [
    [ 7,   0.27, 0.36, 20.7, 0.045, 45, 170, 1.001,  3,    0.45,  8.8, 6 ],
    [ 6.3, 0.3,  0.34,  1.6, 0.049, 14, 132, 0.994,  3.3,  0.49,  9.5, 6 ],
    [ 8.1, 0.28, 0.4,   6.9, 0.05,  30,  97, 0.9951, 3.26, 0.44, 10.1, 6 ]
  ]
}
```
