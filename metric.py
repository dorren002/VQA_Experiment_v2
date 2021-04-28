import h5py
from collections import defaultdict

def compute_tdiuc_accuracy(PATH, preds):
    path_ = f'{PATH}/val_1w_tdiuc.h5'
    gt_answers = h5py.File(path_)['aidx'][:]
    gt_qids = h5py.File(path_)['qid'][:]
    gt_qtypes = h5py.File(path_)['qtypeidx'][:]

    qid2qtype = {qid: gt for qid, gt in zip(gt_qids, gt_qtypes)}
    qid2gt = {qid: gt for qid, gt in zip(gt_qids, gt_answers)}

    acc = defaultdict(list)

    for qid in qid2gt:
        gt = qid2gt[qid]
        qtype = qid2qtype[qid]
        if gt == preds[str(qid)]:
            acc['overall'].append(1)
            acc[qtype].append(1)
        else:
            acc['overall'].append(0)
            acc[qtype].append(0)

    mpt = 0
    overall = 0
    for k in acc:
        if k == 'overall':
            overall = sum(acc[k]) / len(acc[k])
        else:
            mpt += sum(acc[k]) / len(acc[k])
    mpt = mpt / 12

    return mpt, overall, acc


def compute_accuracy(path, dataset, preds):
    mpt, overall,acc = compute_tdiuc_accuracy(path, preds)
    print(f"Mean Per Type: {mpt}, Overall: {overall}\n\n")
    for k in acc:
        if k==overall:
            pass
        else:
            tmp = sum(acc[k]) / len(acc[k])
            print(f"acc on {k} : {tmp}")
