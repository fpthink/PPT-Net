import os
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

base_path = "/test/dataset/benchmark_datasets/"

runs_folder = "oxford/"
filename = "pointcloud_locations_20m_10overlap.csv"
pointcloud_fols = "/pointcloud_20m_10overlap/"

all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))

print(all_folders)
print('data len: {}'.format(len(all_folders)))

folders = []

# All runs are used for training (both full and partial)
index_list = range(len(all_folders))


print("Number of runs: "+str(len(index_list)))
for index in index_list:
    folders.append(all_folders[index])
print(folders)

#####For training and test data split#####
x_width = 150
y_width = 150
p1 = [5735712.768124,620084.402381]
p2 = [5735611.299219,620540.270327]
p3 = [5735237.358209,620543.094379]
p4 = [5734749.303802,619932.693364]
p = [p1,p2,p3,p4]


def check_in_test_set(northing, easting, points, x_width, y_width):
    in_test_set = False
    for point in points:
        if(point[0]-x_width < northing and northing < point[0]+x_width and point[1]-y_width < easting and easting < point[1]+y_width):
            in_test_set = True
            break
    return in_test_set

def construct_query_dict(df_centroids, filename):
    # df_centroids: Nx2
    tree = KDTree(df_centroids[['northing','easting']])
    ind_nn = tree.query_radius(df_centroids[['northing','easting']], r=10)
    ind_r = tree.query_radius(df_centroids[['northing','easting']], r=50)
    queries = {}
    for i in range(len(ind_nn)):
        query = df_centroids.iloc[i]["file"]
        r"""
        >>> a = np.array([1, 2, 3, 2, 4, 1])
        >>> b = np.array([3, 4, 5, 6])
        >>> np.setdiff1d(a, b)
        array([1, 2])
        """
        positives = np.setdiff1d(ind_nn[i], [i]).tolist()   # np.setdiff1d: get difference   less than 10m
        negatives = np.setdiff1d(df_centroids.index.values.tolist(), ind_r[i]).tolist()     # far from 50m
        random.shuffle(negatives)
        queries[i] = {"query":query, "positives":positives, "negatives":negatives}

    with open(filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)


# Initialize pandas DataFrame
df_train = pd.DataFrame(columns=['file','northing','easting'])
df_test = pd.DataFrame(columns=['file','northing','easting'])

for folder in folders:
    df_locations = pd.read_csv(os.path.join(base_path, runs_folder, folder, filename), sep=',')
    df_locations['timestamp'] = runs_folder+folder + pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
    df_locations = df_locations.rename(columns={'timestamp':'file'})

    for index, row in df_locations.iterrows():
        if(check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
            df_test = df_test.append(row, ignore_index=True)
        else:
            df_train = df_train.append(row, ignore_index=True)

print("Number of training submaps: "+str(len(df_train['file'])))
print("Number of non-disjoint test submaps: "+str(len(df_test['file'])))

construct_query_dict(df_train,"training_queries_baseline_v1.pickle")
construct_query_dict(df_test,"test_queries_baseline_v1.pickle")
