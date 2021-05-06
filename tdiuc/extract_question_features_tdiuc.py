import json, h5py
import numpy as np

DATA_PATH = '/home/qzhb/dorren/VQA_Experiment/data'
DATASET = 'TDIUC' # or VQAv2

for split in ['train','val']:
    # Load questions.
    with open(f'{DATA_PATH}/{DATASET}/Questions/{split}_{DATASET}_questions.json','r')as f:
        dt=json.load(f)['questions']
    questions = [q['question'] for q in dt]
    qids = [q['question_id'] for q in dt]
    qids = np.int64(qids)
    del dt

    # Feature file.
    feat_h5 = h5py.File(f'{DATA_PATH}/{DATASET}/questions_{DATASET}_{split}.h5', 'w')
    dt = h5py.special_dtype(vlen=str)
    feat_h5.create_dataset('feats', (len(qids), 2048), dtype=np.float32)
    feat_h5.create_dataset('qids', (len(qids),), dtype=np.int64)
    feat_h5.create_dataset('questions', (len(qids),), dtype=dt)
    feat_h5['qids'][:] = qids
    feat_h5['questions'][:] = questions

    feat_h5.close()
