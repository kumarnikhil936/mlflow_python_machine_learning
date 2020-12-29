import click
import mlflow
import mlflow.keras
import utils

utils.display_versions()


@click.command()
@click.option("--model_uri", help="Model URI", default="mlruns/1/6af605704a7f44b989f3628c5b9289ba/artifacts/keras-model-tf",
              type=str)
@click.option("--data_path", help="Data path",
              default=None, type=str)
def main(model_uri, data_path):
    print("Options:")
    for k, v in locals().items():
        print(f"  {k}: {v}")

    print("\n**** mlflow.keras.load_model\n")
    model = mlflow.keras.load_model(model_uri)
    print("model:", type(model))

    data = utils.get_prediction_data(data_path)
    print("data.type:", type(data))
    print("data.shape:", data.shape)

    print("\n== model.predict")
    predictions = model.predict(data)
    print("predictions.type:", type(predictions))
    print("predictions.shape:", predictions.shape)
    utils.display_predictions(predictions)

    print("\n== model.predict_classes")
    predictions = model.predict_classes(data)
    print("predictions.type:", type(predictions))
    print("predictions.shape:", predictions.shape)
    utils.display_predictions(predictions)

    utils.predict_pyfunc(model_uri, data)


if __name__ == "__main__":
    main()
