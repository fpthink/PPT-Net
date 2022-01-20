import argparse
import importlib
import logging
import os
import time
import sys

import numpy as np
from sklearn.neighbors import KDTree, NearestNeighbors

import util.pointnetvlad_loss as PNV_loss
import torch
import torch.nn as nn
from torch.backends import cudnn
from thop import clever_format
from thop import profile

from tqdm import tqdm
import yaml
from util.util import AverageMeter, check_makedirs, plot_point_cloud
from util.loading_pointclouds import get_query_tuple, get_queries_dict, get_sets_dict, load_pc_files, load_pc_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))

import matplotlib
matplotlib.use("Agg")

os.environ["CUDA_VISIBLE_DEVICE"] = '1'

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='configs/xxx.yaml', required=True, help='config file')
    parser.add_argument('--save_path', type=str, default='exp/xxx', required=True, help='results save path')
    parser.add_argument('--model_name', type=str, default=None, required=True, help='train model name')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r'))
    cfg["save_path"] = args.save_path
    cfg["model_name"] = args.model_name

    return cfg

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def init():
    global args, logger
    global mname
    global TRAINING_QUERIES, TEST_QUERIES
    global device
    global HARD_NEGATIVES, TRAINING_LATENT_VECTORS, TOTAL_ITERATIONS
    
    args = get_parser()
    logger = get_logger()
    
    mname = args["ARCH"]
    
    global DATABASE_SETS, QUERY_SETS
    DATABASE_SETS = get_sets_dict(args["EVAL_DATABASE_FILE"])
    QUERY_SETS = get_sets_dict(args["EVAL_QUERY_FILE"])

    HARD_NEGATIVES = {}
    TRAINING_LATENT_VECTORS = []
    TOTAL_ITERATIONS = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args["MANUAL_SEED"] is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args["MANUAL_SEED"])
        # np.random.seed(args["MANUAL_SEED"])
        # torch.manual_seed(args["MANUAL_SEED"])
        torch.cuda.manual_seed_all(args["MANUAL_SEED"])
    else:
        print("no seed setting!!!")
    logger.info(args)

def load_pc_data(data, train=True):
    len_data = len(data.keys())
    if train:
        logger.info("train len: {}".format(len_data))
        logger.info("please wait about 14 min!...")
    else:
        logger.info("test len: {}".format(len_data))
        logger.info("please wait about some mins!...")
    pcs = []
    cnt_error = 0
    end = time.time()
    for i in tqdm(range(len_data)):
        pc = load_pc_file(data[i]['query'], args["DATASET_FOLDER"])
        pc = pc.astype(np.float32)
        if pc.shape[0] != 4096:
            cnt_error += 1
            logger.info('error data! idx: {}'.format(i))
            continue
        pcs.append(pc)
    pcs = np.array(pcs)
    spd_time = (time.time() - end)/60.
    if train:
        logger.info('train data: {} load data spend: {:.6f}min'.format(pcs.shape, spd_time))
        logger.info('error train data rate: {}/{}'.format(cnt_error, len_data))
    else:
        logger.info('test data: {} load data spend: {:.6f}min'.format(pcs.shape, spd_time))
        logger.info('error test data rate: {}/{}'.format(cnt_error, len_data))
    return pcs

def load_pc_data_set(data_set):
    pc_set = []
    for i in range(len(data_set)):
        pc = load_pc_data(data_set[i], train=False)
        pc_set.append(pc)
    return pc_set

def main():
    init()
    
    Model = importlib.import_module(args["ARCH"])  # import network module
    logger.info("load {}.py success!".format(args["ARCH"]))
    
    model = Model.Network(param=args)
    model = model.to(device)
    
    model_path = os.path.join(args["save_path"], "saved_model", args["model_name"])
    logger.info("load trained model {}".format(model_path))
    checkpoint = torch.load(model_path)
    saved_state_dict = checkpoint['state_dict']
    model.load_state_dict(saved_state_dict)

    logger.info("=> print model ...")
    logger.info(model)

    tmp = args["model_name"].split('.')[0].split('_')
    epoch_name = tmp[1]+'_'+tmp[2]
    strtime = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))

    test_save_root = os.path.join(args["save_path"], "test/{}/result-{}".format(args['EVAL_DATASET'], epoch_name+'_'+strtime))
    check_makedirs(test_save_root)

    eval(model, test_save_root)

def eval(model, test_save_root):
    r"""
    for evaluate test at each epoch
    """
    global eval_database_set, eval_query_set
    eval_database_set = load_pc_data_set(DATABASE_SETS)
    eval_query_set = load_pc_data_set(QUERY_SETS)

    recall = np.zeros(25)
    count = 0
    similarity = []
    one_percent_recall = []

    DATABASE_VECTORS = []
    QUERY_VECTORS = []

    for i in range(len(DATABASE_SETS)):
        DATABASE_VECTORS.append(get_latent_vectors_for_test(model, DATABASE_SETS[i], eval_database_set[i]))

    for j in range(len(QUERY_SETS)):
        QUERY_VECTORS.append(get_latent_vectors_for_test(model, QUERY_SETS[j], eval_query_set[j]))
    
    tot_lost = []
    for m in range(len(DATABASE_SETS)):
        for n in range(len(QUERY_SETS)):
            if m == n: continue
            pair_recall, pair_similarity, pair_opr, lost_num, for_plot = get_recall(m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            tot_lost.append(lost_num)
            for x in pair_similarity:
                similarity.append(x)

    ave_recall = recall / count

    average_similarity = np.mean(similarity)

    ave_one_percent_recall = np.mean(one_percent_recall)

    lost_mean = np.mean(tot_lost)
    lost_sum = np.sum(tot_lost)

    output_file = os.path.join(test_save_root, "result.txt")
    with open(output_file, "w") as output:
        output.write("Average Recall @N:\n")
        logger.info("Average Recall @N:")
        output.write(str(ave_recall))
        logger.info(str(ave_recall))
        output.write("\n\n")
        logger.info("\n")
        output.write("Average Similarity:\n")
        logger.info("Average Similarity:")
        output.write(str(average_similarity))
        logger.info(str(average_similarity))
        output.write("\n\n")
        logger.info("\n")
        output.write("Average Top 1% Recall:\n")
        logger.info("Average Top 1% Recall:")
        output.write(str(ave_one_percent_recall))
        logger.info(str(ave_one_percent_recall))
        output.write("lost mean: {}\n".format(lost_mean))
        logger.info("lost mean: {}\n".format(lost_mean))
        output.write("lost sum: {}\n".format(lost_sum))
        logger.info("lost sum: {}\n".format(lost_sum))
        
    return ave_one_percent_recall

def rotate_point_cloud(batch_data, deg):
    r""" Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        #-90 to 90
        if deg == 10:
            rotation_angle = ((np.random.uniform()*np.pi) - np.pi/2.0)/9.0
        elif deg == 20:
            rotation_angle = ((np.random.uniform()*np.pi) - np.pi/2.0)/9.0*2.0
        elif deg == 30:
            rotation_angle = ((np.random.uniform()*np.pi) - np.pi/2.0)/3.0
        else:
            print('input deg error')
            exit()
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(
            shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def get_latent_vectors_for_test(model, dict_to_process, data):

    model.eval()
    is_training = False
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num = 1
    q_output = []
    times = 0.0
    cnt = 0

    if args['NUM_POINTS'] != 4096:
        nlist = list(range(4096))

    for q_index in range(len(train_file_idxs)//batch_num):
        file_indices = train_file_idxs[q_index*batch_num : (q_index+1)*(batch_num)]
        queries = data[file_indices]

        if args['NUM_POINTS'] != 4096:
            len_q = queries.shape[0]
            tmp = np.zeros((len_q, args['NUM_POINTS'], 3), dtype=np.float32)
            for i in range(len_q):
                tidx = np.random.choice(nlist, size=args['NUM_POINTS'], replace=False)
                tmp[i, :, :] = queries[i, tidx, :]
            queries = tmp
        # if args['DEGREE'] > 0:
        #     queries = rotate_point_cloud(queries, args['DEGREE'])
        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)
            torch.cuda.synchronize()
            start = time.time()
            out = model(feed_tensor, return_feat=False)
            torch.cuda.synchronize()
            times += time.time() - start
            cnt += 1
        out = out.detach().cpu().numpy()
        out = np.squeeze(out)

        q_output.append(out)
    logger.info("inference time: %f" % (times/cnt))

    q_output = np.array(q_output)
    if(len(q_output) != 0):
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # handle edge case
    index_edge = len(train_file_idxs) // batch_num * batch_num
    if index_edge < len(dict_to_process.keys()):
        file_indices = train_file_idxs[index_edge:len(dict_to_process.keys())]
        queries = data[file_indices]

        if args['NUM_POINTS'] != 4096:
            len_q = queries.shape[0]
            tmp = np.zeros((len_q, args['NUM_POINTS'], 3), dtype=np.float32)
            for i in range(len_q):
                tidx = np.random.choice(nlist, size=args['NUM_POINTS'], replace=False)
                tmp[i, :, :] = queries[i, tidx, :]
            queries = tmp

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)
            o1 = model(feed_tensor)

        output = o1.detach().cpu().numpy()
        output = np.squeeze(output)
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    model.train()
    return q_output

def check(idx, idx2, n, m):
    rx = QUERY_SETS[n][idx]['easting']
    ry = QUERY_SETS[n][idx]['northing']

    tx = DATABASE_SETS[m][idx2]['easting']
    ty = DATABASE_SETS[m][idx2]['northing']

    if (rx-tx)*(rx-tx) + (ry-ty)*(ry-ty) <= args['DIST']*args['DIST']:
        return True
    else:
        return False

def get_recall(m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS):

    database_output = DATABASE_VECTORS[m]
    queries_output = QUERY_VECTORS[n]

    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    for_plot = []       # for plot

    num_evaluated = 0
    for i in range(len(queries_output)):
        true_neighbors = QUERY_SETS[n][i][m]
        if(len(true_neighbors) == 0):
            continue
        qname = QUERY_SETS[n][i]['query']
        
        num_evaluated += 1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]),k=num_neighbors)
        
        for_plot.append(QUERY_SETS[n][i]['easting'])
        for_plot.append(QUERY_SETS[n][i]['northing'])

        flag = False
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if check(i, indices[0][j], n, m) is False:
                    continue
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                for_plot.append(j)
                flag = True
                break

        if flag is False:
            for_plot.append(25)

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    return recall, top1_similarity_score, one_percent_recall, num_evaluated-one_percent_retrieved, for_plot

if __name__ == "__main__":
    main()
