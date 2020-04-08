import data_management as dm
import config

def make_predictions(*,path_to_images) -> float:
    """
    Make a prediction using the saved model pipeline
    """
    #Load data
    #Create a dataframe with columns = ["image","target"]
    #column image contains path to the image

    dataframe = path_to_images
    pipe = dm.load_pipeline_keras()
    predictions = pipe.pipe.predict(dataframe)

    return predictions

