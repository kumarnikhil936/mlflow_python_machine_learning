## Overview
* Keras with TensorFlow 2.x train and predict.
* Dataset: MNIST dataset.
* Demonstrates how to serve model with MLflow scoring server
* Saves model as:
  *  MLflow Keras HD5 flavor 
  *  TensorFlow 2.0 SavedModel format as artifact for TensorFlow Serving
* Options to log parameters and metrics.


## MLFlow Server

Run the mlflow server using to access the UI
```
mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 0.0.0.0
```

## Training

Source: [train.py](train.py).


## Autologging

There are two autologging options:
* keras_autolog - calls mlflow.keras.autolog()
* tensorflow_autolog - calls mlflow.tensorflow.autolog()

Interestingly, they behave differently depending on the TensorFlow version.

| TensorFlow Version | Autolog Method | Params | 
|---|---|---|
| 1x | mlflow.keras.autolog | OK | 
| 1x | mlflow.tensorflow.autolog | none |
| 2x | mlflow.keras.autolog | ModuleNotFoundError: No module named 'keras' | 
| 2x | mlflow.tensorflow.autolog | OK |


```
python train.py --experiment_name keras_mnist --epochs 3 --batch_size 128 --keras_autolog True
```

Autologging will create a model under the name `model`.

Autlogging Parameters:
```
acc
loss
```
Autlogging Metrics:
```
batch_size
class_weight
epochs
epsilon
initial_epoch
learning_rate
max_queue_size
num_layers
optimizer_name
sample_weight
shuffle
steps_per_epoch
use_multiprocessing
validation_freq
validation_split
validation_steps
workers
```


To run with user logging (no autologging).
```
python train.py --experiment_name keras_mnist --epochs 3 --batch_size 128
```
or
```
mlflow run . --experiment-name keras_mnist -P epochs=3 -P batch_size=128
```

## Batch Scoring

### Data

By default the prediction scripts get their data from `tensorflow.keras.datasets.mnist.load_data()`.
To specify another file, use the `data_path` option. 
See get_prediction_data() in [utils.py](utils.py) for details.

The following formats are supported:

* json - standard MLflow [JSON-serialized pandas DataFrames](https://mlflow.org/docs/latest/models.html#local-model-deployment) format.
See example [mnist-mlflow.json](mnist-mlflow.json).
* csv - CSV version of above. 
* npz - Compressed Numpy format.
* png - Raw PNG image.


### Score as Keras flavor 

Score as Keras and Keras flavor.
Source: [predict.py](predict.py).

```
python predict.py --model_uri runs:/7e674524514846799310c41f10d6b99d/keras-model
```

```
**** mlflow.keras.load_model

model.type: <class 'tensorflow.python.keras.engine.sequential.Sequential'>
predictions.type: <class 'numpy.ndarray'>
predictions.shape: (10000, 10)
+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+
|           0 |           1 |           2 |           3 |           4 |           5 |           6 |           7 |           8 |           9 |
|-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------|
| 3.123e-06   | 2.60792e-07 | 0.000399815 | 0.000576044 | 3.31058e-08 | 1.12318e-05 | 1.5746e-09  | 0.998986    | 9.80188e-06 | 1.36477e-05 |
| 1.27407e-06 | 5.95377e-05 | 0.999922    | 3.0263e-06  | 6.65168e-13 | 6.7665e-06  | 6.27953e-06 | 1.63278e-11 | 1.39965e-06 | 4.86269e-12 |
.  . .
| 4.17418e-07 | 6.36174e-09 | 8.52869e-07 | 1.0931e-05  | 0.0288905   | 2.07351e-06 | 6.78868e-08 | 0.000951144 | 0.00079286  | 0.969351    |
+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+

**** mlflow.pyfunc.load_model

model.type: <class 'mlflow.pyfunc.PyFuncModel'>
predictions.type: <class 'pandas.core.frame.DataFrame'>
predictions.shape: (10000, 10)
+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+
|           0 |           1 |           2 |           3 |           4 |           5 |           6 |           7 |           8 |           9 |
|-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------|
| 3.123e-06   | 2.60792e-07 | 0.000399815 | 0.000576044 | 3.31058e-08 | 1.12318e-05 | 1.5746e-09  | 0.998986    | 9.80188e-06 | 1.36477e-05 |
| 1.27407e-06 | 5.95377e-05 | 0.999922    | 3.0263e-06  | 6.65168e-13 | 6.7665e-06  | 6.27953e-06 | 1.63278e-11 | 1.39965e-06 | 4.86269e-12 |
.  . .
| 4.17418e-07 | 6.36174e-09 | 8.52869e-07 | 1.0931e-05  | 0.0288905   | 2.07351e-06 | 6.78868e-08 | 0.000951144 | 0.00079286  | 0.969351    |
+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+
```


## Real-time Scoring

Two real-time scoring server solutions are shown here:
* MLflow scoring server
* TensorFlow Servering scoring server

### Real-time Scoring Data

Scoring request data is generated from the reshaped MNIST data saved as JSON data per each scoring server's format.
Create JSON file using [create_scoring_datafiles.py](create_scoring_datafiles.py) file:
```
python create_scoring_files.py --rows 1
```

MLflow scoring server - [mnist-mlflow.json](mnist-mlflow.json)


### Real-time Scoring - MLflow

You can launch the the MLflow scoring server on Local web server 

Data: [mnist-mlflow.json](mnist-mlflow.json)

#### Local Web server

```
mlflow models serve -m mlruns/1/4b5d772f8f2347bea74676d0b50d6b2b/artifacts/keras-model-h5 -h 0.0.0.0 -p 5001
```


#### Score

##### Score JSON file.
```
curl -X POST -H "Content-Type:application/json" -d @./mnist-mlflow.json http://localhost:5001/invocations
```
```
[
  {
    "0": 3.122993575743749e-06,
    "1": 2.6079169401782565e-07,
    "2": 0.0003998146567028016,
    "3": 0.0005760430940426886,
    "4": 3.3105706620517594e-08,
    "5": 1.1231797543587163e-05,
    "6": 1.5745946768674912e-09,
    "7": 0.9989859461784363,
    "8": 9.801864507608116e-06,
    "9": 1.3647688319906592e-05
  },
. . .
]
```

