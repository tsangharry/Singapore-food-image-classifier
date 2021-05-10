from .train import FoodClassifier
import argparse
import logging
from polyaxon_client.tracking import Experiment, get_log_level

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("polyaxon")

    experiment = Experiment()
    logger.info("Starting experiment")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_base', default="Xception")
    parser.add_argument('--base_model_trainable', default=False)
    parser.add_argument('--dense_activation', default=128)
    args = parser.parse_args()

    data_path = '/polyaxon-data/aiap7/workspace/harry_tsang/assignment5/img-classification'
    #data_path = 'img-classification'

    foodclassifier = FoodClassifier(data_path)

    logger.info("loading data")
    foodclassifier.load_data()

    logger.info("creating model")
    foodclassifier.create_model(model_base=args.model_base,
                                dense_activation=int(args.dense_activation))

    logger.info("training model")
    history = foodclassifier.train_model()

    logger.info("evaluating model")
    loss, accuracy = foodclassifier.evaluate_model()

    experiment.log_metrics(loss=loss)
    experiment.log_metrics(accuracy=accuracy)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")

    logger.info("saving h5 and json file")
    foodclassifier.save_model('polyaxon-data/aiap7/workspace/harry_tsang/models/')
