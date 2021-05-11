import time
import os,sys,shutil,argparse
import csv,json,h5py

import torch

from vqa_model import UpDown_CNN_frozed, UpDown
import configs.config_TDIUC_streaming as config
from vqa_dataloader import build_dataloaders, build_rehearsal_dataloader, build_base_init_dataloader, build_rehearsal_dataloader_with_limited_buffer, build_original_dataloader_with_limited_buffer


import vqa_dataloader as vqaloader
from metric import compute_accuracy


parser = argparse.ArgumentParser()
parser.add_argument('--config_name', required=True, type=str)
parser.add_argument('--expt_name', required=True, type=str)

parser.add_argument('--full', action='store_true')

parser.add_argument('--offline', action='store_true')
parser.add_argument('--stream', action="store_true")
parser.add_argument('--remind_original_data', action='store_true')
parser.add_argument('--remind_features', action='store_true')
parser.add_argument('--remind_compressed_features', action='store_true')

parser.add_argument('--data_order', type=str, choices=['iid','qtype'])
parser.add_argument('--rehearsal_mode', type=str, choices=['default','limited_buffer'])
parser.add_argument('--max_buffer_size', type=int, default=None)
parser.add_argument('--sampling_method', type=str, default='random')
parser.add_argument('--buffer_replacement_strategy', type=str, choices=['queue','random'], default='random')
parser.add_argument('--lr', type=float, default=None)

args = parser.parse_args()

# 防覆盖
def assert_expt_name_not_present(expt_dir):
    if os.path.exists(expt_dir):
        raise RuntimeError('Experiment directory {} already exists!'.format(expt_dir))

def inline_print(text):
    print('\r' + text, end="")

def train_epoch(net, criterion, optimizer, data, epoch, net_running):
    net.train()
    total,total_loss = 0,0
    correct, correct_vqa = 0,0
    for qfeat, qseq, imfeat, qid, iid, aidx, ten_aidx, qlen, mfeat in data:
        if config.soft_targets:
            aidx = aidx.cuda()
        else:
            aidx = aidx.long().cuda()

        qlen = qlen.cuda()
        q = qseq.cuda()

        if args.remind_original_data:
            im = image.cuda()
            p = net(q, im, qlen)
        else:
            imfeat = imfeat.cuda()
            p = net(q, imfeat, qlen)

        loss = criterion(p, aidx)
        total_loss += loss*len(qid)
        _, idx = p.max(dim=1)

        if config.soft_targets:
            loss *= config.num_classes
        else:
            exact_match = torch.sum(idx==aidx.long().cuda()).item()
            correct += exact_match

        total += len(qid)
        _, idx = p.max(dim=1, keepdim=True)
        ten_idx = ten_aidx.long().cuda()
        agreeing = torch.sum(ten_idx == idx, dim=1)
        vqa_score = torch.sum((agreeing.type(torch.float32)*0.3).clamp(max=1))
        correct_vqa += vqa_score.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        inline_print('Processed {0} of {1}, Loss:{2:.4f} Accuracy: {3:.4f}, VQA Accuracy: {4:.4f}'.format(
            total,
            len(data.dataset),
            total_loss / total,
            correct / total,
            correct_vqa / total))

        assert vqa_score <= len(qid)
        assert len(qid) <= data.batch_size

    epoch_acc = correct / total
    epoch_vqa_acc = correct_vqa / total
    print('Epoch {}, Accuracy: {}'.format(epoch, epoch_acc))
    print('Epoch {}, VQA Accuracy: {}\n'.format(epoch, epoch_vqa_acc))
    return epoch_acc, epoch_vqa_acc


def training_loop(config, net, train_data, val_data, optimizer, criterion, expt_name, net_running, start_epoch=0):
    eval_net = net_running if config.use_exponential_averaging else net
    for epoch in range(start_epoch, config.max_epochs):
        epoch = epoch + 1
        acc, vqa_acc = train_epoch(net, criterion, optimizer, train_data, epoch, net_running)

        if epoch%config.test_interval == 0:
            acc, vqa_acc = predict(eval_net, val_data, epoch, config.expt_dir, config)

    acc, vqa_acc = predict(eval_net, val_data, epoch, config.expt_dir, config)
    save(eval_net, optimizer, epoch, config.expt_dir, suffix="epoch_"+str(epoch))
    return acc,vqa_acc


def predict(eval_net, data, epoch, expt_name , config, iter_cnt=None):
    print('Testing...')
    eval_net.eval()

    correct, correct_vqa, total = 0, 0, 0
    results = {}
    for qfeat, qseq, imfeat, qid, iid, aidx, ten_aidx, qlen, mfeat in data:
        qlen = qlen.cuda()
        q = qseq.cuda()

        if args.remind_original_data:
            im = image.cuda()
            p = eval_net(q, im, qlen)
        else:
            imfeat = imfeat.cuda()
            p = eval_net(q, imfeat, qlen)

        _, idx = p.max(dim=1)

        if config.soft_targets:
            pass
        else:
            exact_match = torch.sum(idx == aidx.long().cuda()).item()
            correct += exact_match
        total += len(qid)
        _, idx = p.max(dim=1, keepdim=True)
        ten_idx = ten_aidx.long().cuda()
        agreeing = torch.sum(ten_idx == idx, dim=1)
        vqa_score = torch.sum((agreeing.type(torch.float32) * 0.3).clamp(max=1))
        correct_vqa += vqa_score.item()
        inline_print('Processed {0:} of {1:}'.format(
            total,
            len(data) * data.batch_size, ))

        for qqid, pred in zip(qid, idx):
            qqid = str(qqid.item())
            if qqid not in results:
                results[qqid] = int(pred.item())

    if iter_cnt is None:
        fname = 'results_{}_{}_{}_{}.json'.format(data.dataset.split, epoch, config.only_first_k["train"],
                                                  config.data_subset)
    else:
        fname = 'results_ep_{}_{}_{}_{}_iter_{}.json'.format(data.dataset.split, epoch, config.only_first_k["train"],
                                                             config.data_subset, iter_cnt)
    rfile = os.path.join(expt_name, fname)
    json.dump(results, open(rfile, 'w'))
    compute_accuracy(config.data_path, config.dataset, results)
    epoch_acc = correct / total
    epoch_vqa_acc = correct_vqa / total
    with open(os.path.join(expt_name, 'train_log.csv'), mode='a') as log:
        log = csv.writer(log, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        log.writerow([config.only_first_k['train'], config.data_subset, epoch, epoch_acc, epoch_vqa_acc])

    print("\n")
    return epoch_acc, epoch_vqa_acc


def save(net, optimizer, epoch, expt_dir, suffix):
    curr_epoch_path = os.path.join(expt_dir, suffix+'pth')
    latest_path = os.path.join(expt_dir, 'latest.pth')
    data = {'model_state_dict': net.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr']}
    torch.save(data, curr_epoch_path)
    torch.save(data, latest_path)

# 下标s
def get_boundaries(train_data, config):
    data = train_data.dataset.data
    num_pts = len(data)

    boundaries = []
    if config.arrangement['train'] != 'random':
        arr_idxs = [data[idx][config.arrangement['train']] for idx in range(num_pts)]
        cur_idx = arr_idxs[0]
        for idx, a in enumerate(arr_idxs):
            if a != cur_idx:
                cur_idx = a
                boundaries.append(idx)
    elif config.arrangement['train'] == 'random':
        for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            boundaries.append(int(num_pts * i))
    boundaries.append(num_pts)
    return boundaries


def get_base_init_loader(config, train_data):
    boundaries = get_boundaries(train_data, config)
    base_init_ixs = range(0, boundaries[0])
    base_init_data_loader = build_base_init_dataloader(train_data.dataset, base_init_ixs, config.train_batch_size)
    return base_init_ixs, base_init_data_loader

# 网络初始化
def train_base_init(config, net, train_data, val_data, optimizer, criterion, expt_name, net_running):
    base_init_ixs, base_init_data_loader = get_base_init_loader(config, train_data)
    print("\nPerforming base init on {} data points".format(len(base_init_ixs)))

    training_loop(config, net, base_init_data_loader, val_data, optimizer, criterion, expt_name, net_running)
    print("Base init completed!\n")


def merge_data(Qs, Im, Ql, Ai, Qs_r, Im_r, Ql_r, Ai_r):
    data_size = Qs.shape[0] + Qs_r.shape[0]

    Qs_all = torch.zeros((data_size, Qs.shape[1])).long().cuda()
    Qs_all[0] = Qs.squeeze()
    Qs_all[1:] = Qs_r.squeeze().cuda().clone()

    if len(Im.shape) == 3:
        Im_all = torch.zeros((data_size, Im.shape[1], Im.shape[2])).cuda()
    else:
        Im_all = torch.zeros((data_size, Im.shape[1], Im.shape[2], Im.shape[3])).cuda()
    Im_all[0] = Im
    Im_all[1:] = Im_r.clone()

    Ql_all = torch.zeros((data_size)).long().cuda()
    Ql_all[0] = Ql
    Ql_all[1:] = Ql_r.clone()

    Ai_all = torch.zeros((data_size)).long().cuda()
    Ai_all[0] = Ai
    Ai_all[1:] = Ai_r.clone()

    # Sort tensors in descending order of question length
    _, sorted_ixs = torch.sort(Ql_all, descending=True)
    Qs_all = torch.index_select(Qs_all, 0, sorted_ixs)
    Im_all = torch.index_select(Im_all, 0, sorted_ixs)
    Ql_all = torch.index_select(Ql_all, 0, sorted_ixs)
    Ai_all = torch.index_select(Ai_all, 0, sorted_ixs)
    return Qs_all, Im_all, Ql_all, Ai_all


def stream(net, data, test_data, optimizer, criterion, config, net_running):
    eval_net = net_running if config.use_exponential_averaging else net
    net.train()
    iter_cnt, index = 0,0
    boundaries = get_boundaries(data, config)

    for qfeat, qseq, imfeat, qid, iid, aidx, ten_aidx, qlen, mfeat in data:
        net.train()
        if args.stream:
            if iter_cnt == 0:
                print('Training in streaming fashion...')
                print(' Network will evaluate at: {}'.format(boundaries))
            for Q, Qs, Im, Qid, Iid, Ai, Tai, Ql in zip(qfeat, qseq, imfeat, qid, iid, aidx, ten_aidx, qlen):
                iter_cnt += 1
                Qs = Qs.cuda().unsqueeze(0)
                Ql = Ql.cuda().unsqueeze(0)
                Im = Im.cuda().unsqueeze(0)
                Ai = Ai.long().cuda().unsqueeze(0)

                if config.use_lstm:
                    p = net(Qs, Im, Ql)
                else:
                    Qs = Q.cuda().unsqueeze(0)
                    p = net(Qs, Im, Ql)

                # p = net(Qs, Im, Ql)
                loss = criterion(p, Ai)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if iter_cnt in boundaries:
                    print('{} Boundary reached, evaluating...'.format(iter_cnt))
                    predict(eval_net, test_data, 'NA', config.expt_dir, config, iter_cnt)
                    #      save(eval_net, optimizer, 'NA', config.expt_dir, suffix='boundary_{}'.format(iter_cnt))
                    net.train()

            inline_print('Processed {0} of {1}'.format(iter_cnt, len(data) * data.batch_size))

        elif args.remind_original_data:
            net.train()
            # 初始化索引
            if iter_cnt == 0:
                print('\nStreaming with rehearsal...')
                print(' Network will evaluate at: {}'.format(boundaries))
                rehearsal_ixs = []

                rehearsal_data = build_rehearsal_dataloader_with_limited_buffer(data.dataset,
                                                                                    rehearsal_ixs,
                                                                                    config.num_rehearsal_samples,
                                                                                    args.max_buffer_size,
                                                                                    config.buffer_replacement_strategy,
                                                                                    args.sampling_method)
            for Q, Qs, Im, ImFeat, Qid, Iid, Ai, Tai, Ql in zip(qfeat, qseq, image, imfeat, qid, iid, aidx, ten_aidx, qlen):
                iter_cnt += 1
                Qs = Qs.cuda()
                Ql = Ql.cuda()
                Im = Im.cuda()
                Ai = Ai.long().cuda()  # 排序的answer id

                # rehearsal_ixs.append(index)
                # index是buffer里的id，Ai是排序的answerid
                rehearsal_data.batch_sampler.update_buffer(index, int(Ai))

                # Do not stream until we reach the first boundary point
                if index < boundaries[0]:
                    index += 1
                    continue

                # Start streaming after first boundary point
                rehearsal_data_iter = iter(rehearsal_data)

                Qs, Im, Ql, Ai = Qs.unsqueeze(0), Im.unsqueeze(0), Ql.unsqueeze(0), Ai.unsqueeze(0)  # 当前的
                if index > 0:
                    Q_r, Qs_r, Im_r, Qid_r, Iid_r, Ai_r, Tai_r, Ql_r = next(rehearsal_data_iter)
                    # print(Im_r.shape)
                    Qs_merged, Im_merged, Ql_merged, Ai_merged = merge_data(Qs, Im, Ql, Ai, Qs_r, Im_r, Ql_r, Ai_r)
                else:
                    Qs_merged, Im_merged, Ql_merged, Ai_merged = Qs, Im, Ql, Ai

                # print('here')
                p = net(Qs_merged, Im_merged, Ql_merged)
                loss = criterion(p, Ai_merged)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if iter_cnt in boundaries:
                    print('\n\nBoundary {} reached, evaluating...'.format(iter_cnt))
                    predict(eval_net, test_data, 'NA', config.expt_dir, config, iter_cnt)
                    #    save(eval_net, optimizer, 'NA', config.expt_dir, suffix='boundary_{}'.format(iter_cnt))
                    net.train()
                index += 1
            inline_print('Processed {0} of {1}'.format(iter_cnt, len(data) * data.batch_size))
        elif args.remind_features:
            net.train()
            # 初始化索引
            if iter_cnt == 0:
                print('\nStreaming with rehearsal...')
                print(' Network will evaluate at: {}'.format(boundaries))
                rehearsal_ixs = []

                rehearsal_data = build_rehearsal_dataloader_with_limited_buffer(data.dataset,
                                                                                    rehearsal_ixs,
                                                                                    config.num_rehearsal_samples,
                                                                                    args.max_buffer_size,
                                                                                    config.buffer_replacement_strategy,
                                                                                    args.sampling_method)
            for Q, Qs, ImFeat, Qid, Iid, Ai, Tai, Ql, MFeat in zip(qfeat, qseq, imfeat, qid, iid, aidx, ten_aidx, qlen, mfeat):
                iter_cnt += 1
                Qs = Qs.cuda()
                Ql = Ql.cuda()
                Im = ImFeat.cuda()
                Ai = Ai.long().cuda()  # 排序的answer id

                # rehearsal_ixs.append(index)
                # index是buffer里的id，Ai是排序的answerid
                rehearsal_data.batch_sampler.update_buffer(index, int(Ai))

                # Do not stream until we reach the first boundary point
                if index < boundaries[0]:
                    index += 1
                    continue

                # Start streaming after first boundary point
                rehearsal_data_iter = iter(rehearsal_data)

                Qs, Im, Ql, Ai = Qs.unsqueeze(0), Im.unsqueeze(0), Ql.unsqueeze(0), Ai.unsqueeze(0)  # 当前的
                if index > 0:
                    Q_r, Qs_r, Im_r, Qid_r, Iid_r, Ai_r, Tai_r, Ql_r = next(rehearsal_data_iter)
                    # print(Im_r.shape)
                    Qs_merged, Im_merged, Ql_merged, Ai_merged = merge_data(Qs, Im, Ql, Ai, Qs_r, Im_r, Ql_r, Ai_r)
                else:
                    Qs_merged, Im_merged, Ql_merged, Ai_merged = Qs, Im, Ql, Ai

                # print('here')
                p = net(Qs_merged, Im_merged, Ql_merged)
                loss = criterion(p, Ai_merged)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if iter_cnt in boundaries:
                    print('\n\nBoundary {} reached, evaluating...'.format(iter_cnt))
                    predict(eval_net, test_data, 'NA', config.expt_dir, config, iter_cnt)
                    #    save(eval_net, optimizer, 'NA', config.expt_dir, suffix='boundary_{}'.format(iter_cnt))
                    net.train()
                index += 1
            inline_print('Processed {0} of {1}'.format(iter_cnt, len(data) * data.batch_size))


def main():
    dtset = config.dataset
    # 训练集路径
    if args.remind_compressed_features:
        config.feat_path = f'{config.data_path}/all_{dtset}_pq_{args.data_order}.h5'
    else:
        config.feat_path = f'/media/qzhb/DATA1/yi/dorren/all_{dtset}_features.h5'

    print(args)

    # 训练数据记录文件路径
    config.expt_dir = '../snapshots' + args.expt_name
    config.data_order = args.data_order

    # 数据顺序约束
    if config.data_order == 'iid':
        config.arrangement = {'train' : 'random', 'val' : 'random'}
    else:
        config.arrangement = {'train' : 'qtypeidx', 'val' : 'qtypeidx'}

    # 训练记录输出准备
    if not config.overwrite_expt_dir:
        assert_expt_name_not_present(config.expt_dir)
    if not os.path.exists(config.expt_dir):
        os.makedirs(config.expt_dir)
    with open(os.path.join(config.expt_dir,'train_log.csv'),mode='w')as log:
        log = csv.writer(log, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
        log.writerow(['Num_classes', 'Data_subset', 'Epoch', 'Acc', 'VQAAcc'])

    mem_feat = dict()

    if args.full:
        if config.arrangement['train'] == 'random':
            r = range(10)
        else:
            r = range(5)
    else:
        if config.arrangement['train'] == 'random':
            r = [10 * config.data_subset -1]
        else:
            r = [config.only_first_k["train"]-1]

    # 训练
    for i in r:
        if config.arrangement['train'] == 'random':
            config.data_subset = (i+1)/10
        else:
            config.only_first_k['train'] = i+1

        print("Building Dataloaders !")

        data_type = 1
        if args.remind_original_data:
            data_type = 0

        train_data, val_data = build_dataloaders(config, data_type, mem_feat)

        if args.remind_original_data:
            net = UpDown(config)
        else:
            net = UpDown_CNN_frozed(config)
        net_running = None
        print(net)
        net.cuda()

        start_epoch = 0

        net.ques_encoder.embedding.init_embedding('data/glove6b_init_300d_{}.npy'
                                                  .format(config.dataset))

        if args.lr is not None:
            print(f'Using lr specified in args {args.lr}')
            config.lr = args.lr
        else:
            print(f'Using lr specified in {config.lr}')

        optimizer = config.optimizer([p for p in net.parameters() if p.requires_grad==True], lr=config.lr)

        if config.soft_targets:
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()

        print(json.dumps(args.__dict__, indent=4, sort_keys=True))
        shutil.copy('configs/config_'+args.config_name+'.py', os.path.join(config.expt_dir, 'config_'+args.config_name+'.py'))

        print('TRAINING')
        print(f'CURRENT TIME ====== {time.asctime( time.localtime(time.time()))}')

        if args.offline:
            training_loop(config, net, train_data, val_data, optimizer, criterion, config.expt_dir, net_running, start_epoch)
        elif config.max_epochs>0:
            train_base_init(config, net, train_data, val_data, optimizer, criterion, args.expt_name, net_running)

        stream(net, train_data, val_data, optimizer, criterion, config, net_running)
        print('FINISHED')
        print(f'CURRENT TIME ====== {time.asctime(time.localtime(time.time()))}')


if __name__ == '__main__':
    main()

