import logging
import numpy as np
import azure.functions as func
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import json



def user(dfs_user_art, x):
    user = dfs_user_art.loc[dfs_user_art['user_id']==x]
    if len(user)>0:
        return user
    return np.nan


def calcul_cosine_similarity(art_embed):
    cosine_sim = cosine_similarity(art_embed, art_embed)
    return cosine_sim


def arts_recommendations(arts, art_embed, x):
    indices = pd.Series(arts.index, index=arts[0])
    idx = indices[x]
    cosine_sim = calcul_cosine_similarity(art_embed)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    movie_indices = [i[0] for i in sim_scores]
    return sim_scores#x.iloc[movie_indices]


def user_recommendation(dfs_user_art, arts,art_embed, x):
    reco = []
    user_arts = user(dfs_user_art, x)
    for art in user_arts["click_article_id"]:
        livre = arts_recommendations(arts,art_embed, art)
        reco.append(livre[0])
    sim_scores = sorted(reco, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    return sim_scores


def main(req: func.HttpRequest, dfsblob: func.InputStream, dfsuserartblob: func.InputStream, articlesembedblob: func.InputStream) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    # Reading from the input binding
    dfs = bytearray(dfsblob.read())
    dfs_user_art = bytearray(dfsuserartblob.read())
    df_arts_embedd_acp = bytearray(articlesembedblob.read())
    del df_arts_embedd_acp["Unnamed: 0"]
    arts_embedd_acp = df_arts_embedd_acp[[ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
       '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22',
       '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
       '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46',
       '47', '48', '49', '50', '51']]
    arts_embedd_acp = arts_embedd_acp[[ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
       '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22',
       '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
       '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46',
       '47', '48', '49', '50', '51']].to_numpy(dtype = 'float32')

    arts = dfs["click_article_id"].value_counts().index

    name = req.params.get('id_user')

    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            id_user = req_body.get('id_user')
            result = user_recommendation(dfs_user_art,arts,arts_embedd_acp,id_user)
            result = result.to_json(orient="split")
            return func.HttpResponse(
                                    json.dumps(result),
                                    mimetype="application/json",
                                )
    if name:
        result = user_recommendation(dfs_user_art,arts,arts_embedd_acp,name)
        result = result.to_json(orient="split")
        return func.HttpResponse(
                                    json.dumps(result),
                                    mimetype="application/json",
                                )
        #return func.HttpResponse(f"Hello, {str(id_user)}. This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
