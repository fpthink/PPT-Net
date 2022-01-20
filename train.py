import argparse
import importlib
import logging
import os
import time
import sys


import numpy as np
from sklearn.neighbors import KDTree, NearestNeighbors
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

import util.pointnetvlad_loss as PNV_loss
import torch
import torch.nn as nn
from torch.backends import cudnn
from thop import clever_format
from thop import profile

from tqdm import tqdm
import yaml
import random
from util.util import AverageMeter, check_makedirs, count_files
from util.loading_pointclouds import get_query_tuple, get_queries_dict, get_sets_dict, load_pc_files, load_pc_file
from util.loading_pointclouds import TrainTransform, TrainSetTransform

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='configs/pvcnnvlad.yaml', help='config file')
    parser.add_argument('--save_path', type=str, default='exp/test', help='evaluate')
    parser.add_argument('--weight', type=str, default=None, help='weight')
    parser.add_argument('--resume', type=str, default=None, help='resume')
    parser.add_argument('--eval', default=False, action='store_true', help='evaluation of the model')

    args = parser.parse_args()
    print('args.config: {}'.format(args.config))
    cfg = yaml.safe_load(open(args.config, 'r'))
    cfg["save_path"] = args.save_path
    cfg["weight"] = args.weight
    cfg["resume"] = args.resume
    cfg["eval"] = args.eval
    
    if cfg["DATA_TYPE"] == "baseline":
        # train_baseline = "training_queries_baseline_v1.pickle"
        # test_baseline = "test_queries_baseline_v1.pickle"
        train_baseline = "training_queries_baseline_short.pickle"
        test_baseline = "test_queries_baseline_short.pickle"
    elif cfg["DATA_TYPE"] == "refine":
        train_baseline = "training_queries_refine_v1.pickle"
        test_baseline = "test_queries_baseline_v1.pickle"
    else:
        logger.info("DATA_TYPE is not support, only support: 'baseline', 'refine', and 'lpd_baseline'")
        exit()

    cfg["TRAIN_FILE"] = os.path.join(cfg["TRAIN_FILE_ROOT"], train_baseline)
    cfg["TEST_FILE"] = os.path.join(cfg["TEST_FILE_ROOT"], test_baseline)

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
    # global variables
    global args, logger
    global TRAINING_QUERIES, TEST_QUERIES
    global device
    global HARD_NEGATIVES, TRAINING_LATENT_VECTORS, TOTAL_ITERATIONS

    args = get_parser()
    logger = get_logger()

    # Load dictionary of training queries
    if not args["eval"]:
        TRAINING_QUERIES = get_queries_dict(args["TRAIN_FILE"])
        # TEST_QUERIES = get_queries_dict(args["TEST_FILE"])
    
    global DATABASE_SETS, QUERY_SETS
    DATABASE_SETS = get_sets_dict(args["EVAL_DATABASE_FILE"])
    QUERY_SETS = get_sets_dict(args["EVAL_QUERY_FILE"])

    HARD_NEGATIVES = {}
    TRAINING_LATENT_VECTORS = []
    TOTAL_ITERATIONS = 0

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args["TRAIN_GPU"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args["MANUAL_SEED"] is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args["MANUAL_SEED"])
        torch.cuda.manual_seed_all(args["MANUAL_SEED"])
    else:
        print("no seed setting!!!")
    logger.info(args)

def get_bn_decay(batch):
    bn_momentum = args["BN_INIT_DECAY"] * \
        (args["BN_DECAY_DECAY_RATE"] **
         (batch * args["BATCH_NUM_QUERIES"] // args["DECAY_STEP"]))
    return min(args["BN_DECAY_CLIP"], 1 - bn_momentum)

# learning rate halfed every 5 epoch
def get_learning_rate(epoch):
    learning_rate = args["BASE_LEARNING_RATE"] * ((0.9) ** (epoch // 1))
    learning_rate = max(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate

def get_current_lr(optimizer):
    return optimizer.param_groups[0]['lr']

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
        pc = load_pc_file(data[i]['query'], args["DATASET_FOLDER"], 3, args['NUM_POINTS'])
        pc = pc.astype(np.float32)
        if pc.shape[0] != args['NUM_POINTS']:
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

    global train_data, test_data
    if not args["eval"]:
        train_data = load_pc_data(TRAINING_QUERIES, train=True)
        # test_data = load_pc_data(TEST_QUERIES, train=False)

    global eval_database_set, eval_query_set
    eval_database_set = load_pc_data_set(DATABASE_SETS)
    eval_query_set = load_pc_data_set(QUERY_SETS)

    global HARD_NEGATIVES, TOTAL_ITERATIONS
    bn_decay = get_bn_decay(0)

    if args["LOSS_FUNCTION"] == 'quadruplet':
        loss_function = PNV_loss.quadruplet_loss
    elif args["LOSS_FUNCTION"] == 'hphn_quadruplet':
        loss_function = PNV_loss.hphn_quadruplet_loss
    else:
        loss_function = PNV_loss.triplet_loss_wrapper

    # import network module
    Model = importlib.import_module(args["ARCH"])
    logger.info("load {}.py success!".format(args["ARCH"]))

    cmd_str = "cp ./models/{}.py {}".format(args["ARCH"], args["save_path"])
    print("cmd_str: {}".format(cmd_str))
    os.system(cmd_str)

    model = Model.Network(param=args)
    model = model.to(device)
    
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    learning_rate = args["BASE_LEARNING_RATE"]
    if args["OPTIMIZER"] == 'momentum':
        optimizer = torch.optim.SGD(parameters, learning_rate, momentum=args["MOMENTUM"])
    elif args["OPTIMIZER"] == 'adam':
        optimizer = torch.optim.Adam(parameters, learning_rate)
    else:
        optimizer = None
        exit(0)
    
    if args["resume"] is not None:
        resume_filename = os.path.join(args["save_path"], "saved_model", args["resume"])
        logger.info("Resuming From {}".format(resume_filename))
        checkpoint = torch.load(resume_filename)
        saved_state_dict = checkpoint['state_dict']
        starting_epoch = checkpoint['epoch']
        # TOTAL_ITERATIONS = starting_epoch * len(TRAINING_QUERIES)
        model.load_state_dict(saved_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        starting_epoch = 0

    if args["LEARNING_RATE_DECAY"] == 'step':
        lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.2)
    elif args["LEARNING_RATE_DECAY"] == 'cosine':
        lr_scheduler = CosineAnnealingLR(optimizer, args["MAX_EPOCH"], eta_min=learning_rate)
    else:
        lr_scheduler = None

    logger.info("=> creating model ...")
    logger.info(model)
    model = nn.DataParallel(model)
    total = sum([param.nelement() for param in model.parameters()])
    logger.info("Number of parameter: %.2fM" % (total / 1e6))
    
    if args["eval"] is not True:
        train(starting_epoch, model, optimizer, None, loss_function, lr_scheduler)
    else:
        eval(model, starting_epoch)

def train(starting_epoch, model, optimizer, train_writer, loss_function, lr_scheduler):
    for epoch in range(starting_epoch, args["MAX_EPOCH"]):
        logger.info('**** EPOCH {:03d} ****'.format(epoch))
        train_one_epoch(model, optimizer, train_writer, loss_function, epoch)
        # evaluate
        eval_recall = eval(model, epoch)
        if lr_scheduler is not None:
            lr_scheduler.step()

def train_one_epoch(model, optimizer, train_writer, loss_function, epoch):
    global HARD_NEGATIVES
    global TRAINING_LATENT_VECTORS, TOTAL_ITERATIONS

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    inference = AverageMeter()

    is_training = True
    
    sampled_neg = 4000
    
    # number of hard negatives in the training tuple
    # which are taken from the sampled negatives
    num_to_take = 10

    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAINING_QUERIES.keys()))
    np.random.shuffle(train_file_idxs)

    iter_num = len(train_file_idxs)//args["BATCH_NUM_QUERIES"]
    max_iter = args["MAX_EPOCH"] * iter_num
    cur_lr = get_current_lr(optimizer)
    batch_size = args["TRAIN_POSITIVES_PER_QUERY"] * args["TRAIN_NEGATIVES_PER_QUERY"]

    end = time.time()
    for i in range(iter_num):
        batch_keys = train_file_idxs[i*args["BATCH_NUM_QUERIES"] : (i+1)*args["BATCH_NUM_QUERIES"]]
        q_tuples = []

        faulty_tuple = False
        no_other_neg = False
        for j in range(args["BATCH_NUM_QUERIES"]):
            if len(TRAINING_QUERIES[batch_keys[j]]["positives"]) < args["TRAIN_POSITIVES_PER_QUERY"]:
                faulty_tuple = True
                break

            # no cached feature vectors, random generate triplets
            if len(TRAINING_LATENT_VECTORS) == 0:
                q_tuples.append(
                    get_query_tuple(batch_keys[j], TRAINING_QUERIES[batch_keys[j]], args["TRAIN_POSITIVES_PER_QUERY"], args["TRAIN_NEGATIVES_PER_QUERY"],
                                    TRAINING_QUERIES, hard_neg=[], other_neg=True, dataset_folder=args["DATASET_FOLDER"],
                                    data=train_data, num_points=args['NUM_POINTS']))
                    # get_query_tuple(TRAINING_QUERIES[batch_keys[j]], args["TRAIN_POSITIVES_PER_QUERY"], args["TRAIN_NEGATIVES_PER_QUERY"],
                    #                 TRAINING_QUERIES, hard_neg=[], other_neg=True, dataset_folder=args["DATASET_FOLDER"],
                    #                 data=train_data))
                # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_neg=[], other_neg=True, dataset_folder=args["DATASET_FOLDER"]))
                # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_neg=[], other_neg=True, dataset_folder=args["DATASET_FOLDER"]))

            elif batch_keys[j] not in HARD_NEGATIVES:   # mining hard tuples by TRAINING_LATENT_VECTORS
                # query = get_feature_representation(batch_keys[j], TRAINING_QUERIES[batch_keys[j]]['query'], model)      # inference query point cloud
                # np.random.shuffle(TRAINING_QUERIES[batch_keys[j]]['negatives'])
                # negatives = TRAINING_QUERIES[batch_keys[j]]['negatives'][0:sampled_neg]
                query = TRAINING_LATENT_VECTORS[batch_keys[j]]
                # print('{}'.format(len(TRAINING_QUERIES[batch_keys[j]]['negatives'])))
                negatives = np.random.choice(TRAINING_QUERIES[batch_keys[j]]['negatives'], sampled_neg, replace=False)
                hard_negs = get_random_hard_negatives(query, negatives, num_to_take)  # choose 10 hardest negative, spend about 40ms
                # HARD_NEGATIVES[batch_keys[j]] = hard_negs
                q_tuples.append(
                    get_query_tuple(batch_keys[j], TRAINING_QUERIES[batch_keys[j]], args["TRAIN_POSITIVES_PER_QUERY"], args["TRAIN_NEGATIVES_PER_QUERY"],
                                    TRAINING_QUERIES, hard_negs, other_neg=True, dataset_folder=args["DATASET_FOLDER"],
                                    data=train_data, num_points=args['NUM_POINTS']))
                # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True, dataset_folder=args["DATASET_FOLDER"]))
                # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True, dataset_folder=args["DATASET_FOLDER"]))
            else:
                # query = get_feature_representation(batch_keys[j], TRAINING_QUERIES[batch_keys[j]]['query'], model)
                # np.random.shuffle(TRAINING_QUERIES[batch_keys[j]]['negatives'])
                # negatives = TRAINING_QUERIES[batch_keys[j]]['negatives'][0:sampled_neg]
                query = TRAINING_LATENT_VECTORS[batch_keys[j]]
                negatives = np.random.choice(TRAINING_QUERIES[batch_keys[j]]['negatives'], sampled_neg, replace=False)
                negatives = np.concatenate([negatives, HARD_NEGATIVES[batch_keys[j]]],axis=0)
                hard_negs = get_random_hard_negatives(query, negatives, num_to_take)
                # hard_negs = list(set().union(HARD_NEGATIVES[batch_keys[j]], hard_negs)) 
                # random.shuffle(hard_negs)
                # hard_negs = hard_negs[:num_to_take]
                HARD_NEGATIVES[batch_keys[j]] = hard_negs
                q_tuples.append(
                    get_query_tuple(batch_keys[j], TRAINING_QUERIES[batch_keys[j]], args["TRAIN_POSITIVES_PER_QUERY"], args["TRAIN_NEGATIVES_PER_QUERY"],
                                    TRAINING_QUERIES, hard_negs, other_neg=True, dataset_folder=args["DATASET_FOLDER"],
                                    data=train_data, num_points=args['NUM_POINTS']))
                # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True, dataset_folder=args["DATASET_FOLDER"]))
                # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True, dataset_folder=args["DATASET_FOLDER"]))

            if q_tuples[j][3].shape[0] != args["NUM_POINTS"]:
                no_other_neg = True
                break

        if faulty_tuple:
            logger.info('Epoch: [{}/{}][{}/{}] FAULTY TUPLE!!!'.format(epoch, args["MAX_EPOCH"], i+1, iter_num))
            continue

        if no_other_neg:
            logger.info('Epoch: [{}/{}][{}/{}] NO OTHER NEG!!!'.format(epoch, args["MAX_EPOCH"], i+1, iter_num))
            continue

        queries = []
        positives = []
        negatives = []
        other_neg = []
        for k in range(len(q_tuples)):
            queries.append(q_tuples[k][0])
            positives.append(q_tuples[k][1])
            negatives.append(q_tuples[k][2])
            other_neg.append(q_tuples[k][3])

        queries = np.array(queries, dtype=np.float32)
        queries = np.expand_dims(queries, axis=1)
        other_neg = np.array(other_neg, dtype=np.float32)
        other_neg = np.expand_dims(other_neg, axis=1)
        positives = np.array(positives, dtype=np.float32)
        negatives = np.array(negatives, dtype=np.float32)

        if len(queries.shape) != 4:
            logger.info('Epoch: [{}/{}][{}/{}] FAULTY TUPLE!!!'.format(epoch, args["MAX_EPOCH"], i+1, iter_num))
            continue
        
        data_time.update(time.time() - end)

        model.train()
        optimizer.zero_grad()
        start = time.time()
        # torch.cuda.synchronize()
        output_queries, output_positives, output_negatives, output_other_neg = run_model(model, queries, positives, negatives, other_neg)
        # torch.cuda.synchronize()
        inference.update(time.time() - start)
        loss = loss_function(output_queries, output_positives, output_negatives, output_other_neg, 
                            args["MARGIN_1"], args["MARGIN_2"], use_min=args["TRIPLET_USE_BEST_POSITIVES"],
                            lazy=args["LOSS_LAZY"], ignore_zero_loss=args["LOSS_IGNORE_ZERO_BATCH"])
        if loss > 1e-10:
            loss.backward()
            optimizer.step()
        
        loss_meter.update(loss.cpu().item(), batch_size)
        batch_time.update(time.time() - end)
        end = time.time()
        
        TOTAL_ITERATIONS += args["BATCH_NUM_QUERIES"]

        # calculate remain time
        current_iter = epoch * iter_num + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        logger.info('Epoch: [{}/{}][{}/{}] '
                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Remain {remain_time} '
                    'Loss {loss_meter.val:.4f} '
                    'lr {lr:.10f} '.format(epoch, args["MAX_EPOCH"], i+1, iter_num,
                                            batch_time=inference, data_time=data_time,
                                            remain_time=remain_time,
                                            loss_meter=loss_meter,
                                            lr=cur_lr))
        if epoch > 5 and i % (1400 // args["BATCH_NUM_QUERIES"]) == 29:
            TRAINING_LATENT_VECTORS = get_latent_vectors(model, TRAINING_QUERIES)
            logger.info("Updated cached feature vectors")
        if i % (6000 // args["BATCH_NUM_QUERIES"]) == 101:
            save_model(model, epoch, optimizer, i)
    save_model(model, epoch, optimizer)

def eval(model, epoch):
    r"""
    for evaluate test at each epoch
    """
    test_save_root = os.path.join(args["save_path"], "test/result_epoch_{}".format(str(epoch)))
    check_makedirs(test_save_root)

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
    
    for m in range(len(QUERY_SETS)):
        for n in range(len(QUERY_SETS)):
            if m == n: continue
            pair_recall, pair_similarity, pair_opr = get_recall(m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x)

    ave_recall = recall / count

    average_similarity = np.mean(similarity)

    ave_one_percent_recall = np.mean(one_percent_recall)

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
    return ave_one_percent_recall

def get_latent_vectors_for_test(model, dict_to_process, data):
    model.eval()
    is_training = False
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num = args["EVAL_BATCH_SIZE"] * (1 + args["EVAL_POSITIVES_PER_QUERY"] + args["EVAL_NEGATIVES_PER_QUERY"])
    q_output = []
    times = 0.0
    cnt = 0

    if args['NUM_POINTS'] < 4096:
        nlist = list(range(4096))

    for q_index in range(len(train_file_idxs)//batch_num):
        file_indices = train_file_idxs[q_index*batch_num : (q_index+1)*(batch_num)]
        queries = data[file_indices]
        if args['NUM_POINTS'] < 4096:
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
            torch.cuda.synchronize()
            start = time.time()
            out = model(feed_tensor)
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
        if args['NUM_POINTS'] < 4096:
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

def get_recall(m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS):

    database_output = DATABASE_VECTORS[m]
    queries_output = QUERY_VECTORS[n]

    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        true_neighbors = QUERY_SETS[n][i][m]
        if(len(true_neighbors) == 0):
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(
            np.array([queries_output[i]]),k=num_neighbors)
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    return recall, top1_similarity_score, one_percent_recall

def save_model(model, epoch, optimizer, i=None):
    if isinstance(model, nn.DataParallel):
        model_to_save = model.module
    else:
        model_to_save = model

    if i is not None:
        save_name = os.path.join(args["save_path"], "saved_model/train_epoch_{}_iter{}.pth".format(str(epoch), str(i)))
    else:
        save_name = os.path.join(args["save_path"], "saved_model/train_epoch_{}_end.pth".format(str(epoch)))
    torch.save({
        'epoch': epoch,
        'iter': TOTAL_ITERATIONS,
        'state_dict': model_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, save_name)
    logger.info("Model Saved As {}".format(save_name))

def get_feature_representation(idx, filename, model):
    model.eval()
    idxs = [idx]
    queries = train_data[idxs]
    queries = np.expand_dims(queries, axis=1)
    
    with torch.no_grad():
        q = torch.from_numpy(queries).float()
        q = q.to(device)
        output = model(q)

    output = output.detach().cpu().numpy()
    output = np.squeeze(output)
    model.train()
    return output

def get_random_hard_negatives(query_vec, random_negs, num_to_take):
    global TRAINING_LATENT_VECTORS

    latent_vecs = []
    latent_vecs = TRAINING_LATENT_VECTORS[random_negs]

    latent_vecs = np.array(latent_vecs)
    nbrs = KDTree(latent_vecs)
    distances, indices = nbrs.query(np.array([query_vec]), k=num_to_take)   # select the nearest 10 as the hardest samples
    hard_negs = np.squeeze(np.array(random_negs)[indices[0]])
    hard_negs = hard_negs.tolist()
    return hard_negs

def get_latent_vectors(model, dict_to_process):
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num = args["BATCH_NUM_QUERIES"] * (1 + args["TRAIN_POSITIVES_PER_QUERY"] + args["TRAIN_NEGATIVES_PER_QUERY"] + 1)
    q_output = []

    model.eval()

    if args['NUM_POINTS'] < 4096:
        nlist = list(range(4096))

    for q_index in range(len(train_file_idxs)//batch_num):
        file_indices = train_file_idxs[q_index*batch_num : (q_index+1)*(batch_num)]
        queries = train_data[file_indices]
        if args['NUM_POINTS'] < 4096:
            len_q = queries.shape[0]
            tmp = np.zeros((len_q, args['NUM_POINTS'], 3), dtype=np.float32)
            for i in range(len_q):
                tidx = np.random.choice(nlist, size=args['NUM_POINTS'], replace=False)
                tmp[i, :, :] = queries[i, tidx, :]
            queries = tmp

        feed_tensor = torch.from_numpy(queries).float()
        feed_tensor = feed_tensor.unsqueeze(1)
        feed_tensor = feed_tensor.to(device)
        with torch.no_grad():
            out = model(feed_tensor)

        out = out.detach().cpu().numpy()
        out = np.squeeze(out)

        q_output.append(out)

    q_output = np.array(q_output)
    if(len(q_output) != 0):
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # handle edge case
    for q_index in range((len(train_file_idxs) // batch_num * batch_num), len(dict_to_process.keys())):
        index = [train_file_idxs[q_index]]
        queries = train_data[index]

        if args['NUM_POINTS'] < 4096:
            len_q = queries.shape[0]
            tmp = np.zeros((len_q, args['NUM_POINTS'], 3), dtype=np.float32)
            for i in range(len_q):
                tidx = np.random.choice(nlist, size=args['NUM_POINTS'], replace=False)
                tmp[i, :, :] = queries[i, tidx, :]
            queries = tmp

        queries = np.expand_dims(queries, axis=1)

        with torch.no_grad():
            queries_tensor = torch.from_numpy(queries).float()
            o1 = model(queries_tensor)

        output = o1.detach().cpu().numpy()
        output = np.squeeze(output)
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    model.train()
    return q_output

def run_model(model, queries, positives, negatives, other_neg, require_grad=True):
    queries_tensor = torch.from_numpy(queries).float()
    positives_tensor = torch.from_numpy(positives).float()
    negatives_tensor = torch.from_numpy(negatives).float()
    other_neg_tensor = torch.from_numpy(other_neg).float()
    feed_tensor = torch.cat((queries_tensor, positives_tensor, negatives_tensor, other_neg_tensor), 1)
    feed_tensor = feed_tensor.view((-1, 1, args["NUM_POINTS"], 3))
    feed_tensor.requires_grad_(require_grad)
    feed_tensor = feed_tensor.to(device)
    if require_grad:
        output = model(feed_tensor)
    else:
        with torch.no_grad():
            output = model(feed_tensor)
    output = output.view(args["BATCH_NUM_QUERIES"], -1, args["FEATURE_OUTPUT_DIM"])
    o1, o2, o3, o4 = torch.split(output, [1, args["TRAIN_POSITIVES_PER_QUERY"], args["TRAIN_NEGATIVES_PER_QUERY"], 1], dim=1)
    return o1, o2, o3, o4

if __name__ == "__main__":
    main()
