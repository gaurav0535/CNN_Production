import pandas as pd
from glob import glob
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier

import model as m
from config import DATA_FOLDER,PIPELINE_PATH,CLASSES_PATH,MODEL_PATH,BATCH_SIZE,EPOCHS

def load_image_paths(data_folder):
    """
         let's create a dataframe:
         the dataframe stores the path to the image in one column
         and the class of the weed (the target) in the next column
         functions returns the dataframe
    """
    images_df = []

    # navigate within each folder
    for class_folder_name in os.listdir(DATA_FOLDER):
        class_folder_path = os.path.join(DATA_FOLDER, class_folder_name)

        # collect every image path
        for image_path in glob(os.path.join(class_folder_path, "*.png")):
            tmp = pd.DataFrame([image_path, class_folder_name]).T

            print(tmp)
            images_df.append(tmp)

    # concatenate the final df
    images_df = pd.concat(images_df, axis=0, ignore_index=True)
    images_df.columns = ['image', 'target']

    return images_df

def get_train_test_target(df):

    X_train,X_test,y_train,y_test = train_test_split(df['image'],
                                                     df['target'],
                                                     test_size=.20,
                                                     random_state=101)

    #reset index
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train,X_test,y_train,y_test

def save_pipeline_keras(model):

    joblib.dump(model.named_steps["dataset"], PIPELINE_PATH)
    joblib.dump(model.named_steps["cnn_models"].clsses_,CLASSES_PATH)
    model.named_steps['cnn_model'].model.save(MODEL_PATH)

def load_pipeline_keras():
    dataset = joblib.load(PIPELINE_PATH)

    build_model = lambda:load_model(MODEL_PATH)

    classifier = KerasClassifier(build_fn=build_model,
                                 batch_size = BATCH_SIZE,
                                 validation_split = 10,
                                 epochs = EPOCHS,
                                 verbose = 2,
                                 callbacks = m.callbacks_list)

    classifier.classes_ = joblib.load(CLASSES_PATH)
    classifier.model = build_model()

    return Pipeline([
        ('dataset',dataset),
        ('cnn_model',classifier)
    ])



