import torch
import pandas as pd
from sklearn import preprocessing

RAW_FILE_PATH = "../data/movies/raw/ratings.csv"
NEW_FILE_PATH = "../data/movies/processed/ratings.csv"
SAMPLE_SIZE = None

dataset = pd.read_csv(RAW_FILE_PATH)

user_labels = preprocessing.LabelEncoder()
movie_labels = preprocessing.LabelEncoder()

dataset.userId = user_labels.fit_transform(dataset.userId.values)
dataset.movieId = movie_labels.fit_transform(dataset.movieId.values)

num_users = max(dataset.userId) + 1
num_movies = max(dataset.movieId) + 1

if SAMPLE_SIZE is not None:
    dataset = dataset.head(SAMPLE_SIZE)

dataset.to_csv(NEW_FILE_PATH)

def init_interaction_edges(df, src, dest, link, thresh=None):
    # constructs interaction edges
    src_ids = df[src].tolist()
    dest_ids = df[dest].tolist()
    links = torch.tensor(df[link])

    if thresh is not None:
        links_bool = links >= thresh

    locations = [[], []] # src, dest
    values = []

    for i in range(links.shape[0]):
        if links_bool[i]:
            locations[0].append(src_ids[i])
            locations[1].append(dest_ids[i])
            values.append(links[i])

    return torch.LongTensor(locations), torch.tensor(values)