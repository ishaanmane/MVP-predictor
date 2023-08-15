import pandas as pd
from sklearn.linear_model import Ridge
import pickle

stats = pd.read_csv("player_mvp_stats.csv", index_col=0)
stats = stats.fillna(0)
predictors = ["Age", "G", "GS", "MP", "FG", "FGA", 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'W', 'L', 'W/L%',       'GB', 'PS/G', 'PA/G', 'SRS']

model = Ridge(alpha=.1)

years = range(1991, 2021)

def add_ranks(predictions):
    predictions = predictions.sort_values("Predictions", ascending=False)
    predictions["Predicted Rank"] = list(range(1, predictions.shape[0]+1))
    predictions = predictions.sort_values("Share", ascending=False)
    predictions["Rank"] = list(range(1, predictions.shape[0]+1))
    predictions["Diff"] = (predictions["Rank"] - predictions["Predicted Rank"])
    return predictions

def predict_it(stats, model, get_year, predictors):
    all_predictions=[]
    train = stats[stats["Year"] < get_year]
    test = stats[stats["Year"] == get_year]
    model.fit(train[predictors], train["Share"])
    predictions = model.predict(test[predictors])
    predictions = pd.DataFrame(predictions, columns=["Predictions"], index=test.index)
    combination = pd.concat([test[["Player", "Share"]], predictions], axis=1)
    combination = add_ranks(combination)
    all_predictions.append(combination)
    return all_predictions

pickle.dump(model, open("model.pkl", "wb"))
