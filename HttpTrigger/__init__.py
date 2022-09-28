import logging
import numpy as np
import azure.functions as func
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import json
from io import BytesIO
# predicitng
from pprint import pprint as pp
import pickle
import surprise
from azure.storage.blob import BlobClient

def user(dfs_user_art, x):
    logging.info('---1 -------begin user()')
    user = dfs_user_art.loc[dfs_user_art['user_id'] == x]
    if len(user) > 0:
        logging.info('---1 -------end user()')
        return user
    return np.nan

def transform_to_dataframe(dfblob, ityp=0):
    dfs = bytearray(dfblob.read())
    dfs = pd.read_csv(BytesIO(dfs))
    if ityp==1:
        dfs = pd.read_csv(BytesIO(dfs), dtype=np.float32)
    return dfs


def generate_recommendation(model_b, user_id, dfs_user_art, dfs, n_items): 
    logging.info('=========debut funct generatereco')
    logging.info('=========debut pickle')
    model =  pickle.loads(model_b)
    model = model["algo"]
    logging.info('=========end pickle')
    # Obtenir une liste de tous les identifiants de films à partir du jeu de données 
    arts_ids = dfs["click_article_id"].value_counts().index
 
    # Obtenir une liste de tous les identifiants de films qui ont été regardés par l'utilisateur 
    arts_ids_user = user(dfs_user_art, user_id)
    # Obtenir une liste de tous les ID de films qui n'ont pas été regardés par l'utilisateur 
    arts_ids_to_pred = np.setdiff1d(arts_ids, arts_ids_user) 
 
    # Appliquer une note de 4 à toutes les interactions (uniquement pour correspondre au format de l'ensemble de données Surprise) 
    test_set = [[user_id, art_id, 0] for art_id in arts_ids_to_pred] 
    
    # Prédire les notes et générer des recommandations 
    logging.info('=========debut predictions')
    predictions = model.test(test_set)
    logging.info('=========end predictions')
    pred_ratings = np.array([pred.est for pred in predictions]) 
    print("Top {0} recommandations d'articles pour l'utilisateur {1} :".format(n_items, user_id)) 
    # Classer les n meilleurs films en fonction des prédictions notes 
    index_max = (-pred_ratings).argsort()[:n_items] 
    result = []
    for i in index_max: 
        art_id = arts_ids_to_pred[i] 
        print(dfs[dfs["click_article_id"]==art_id]["click_article_id"].values[0] , pred_ratings[i])
        result.append({"article_id":dfs[dfs["click_article_id"]==art_id]["click_article_id"].values[0] , "predictions":str(pred_ratings[i])})
    logging.info('=========end funct generatereco')
    return result


def main(req: func.HttpRequest, dfsblob: func.InputStream) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    blob_client = BlobClient.from_blob_url("https://rgproject9weub2b4.blob.core.windows.net/input/model.pkl?sp=r&st=2022-09-28T04:04:08Z&se=2023-09-28T12:04:08Z&spr=https&sv=2021-06-08&sr=b&sig=B5wtMVnqNKjzodiTotO2ci6Z9OfVCYrEEUuj6pe9nfs%3D")
    download_stream = blob_client.download_blob()
    logging.info('=========below is content of test1')
    model_b = download_stream.readall()
    logging.info('=========above is content of test1')
    logging.info('=========below is content of files csv')
    dfs = transform_to_dataframe(dfsblob)
    dfs_user_art = dfs.groupby(["user_id", "click_article_id"])["click_article_id"].count().reset_index(name="nb_click_by_arts")
    logging.info('=========above is content of files csv')
    req_body_bytes = req.get_body()
    req_body = req_body_bytes.decode("utf-8")
    json_body = json.loads(req_body)
    # logging.info("test:")
    name = None
    name = json_body['id_user']
    logging.info(f"Request name: {name}")
    if name is not None:
        name = int(name)
        logging.info('=========debut generatereco')
        result = generate_recommendation(model_b, name, dfs_user_art, dfs, 5)
        logging.info('=========end generatereco')
        func.HttpResponse.charset = 'utf-8'
        logging.info("------------------------------------------finghgjgjh result ")
        return func.HttpResponse(
                json.dumps(result),
                status_code=200
                )
    else:
        return func.HttpResponse(
              "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
              status_code=200
        )
