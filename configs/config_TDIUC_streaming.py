import torch
from dictionary import Dictionary
import sys

# from s_mac import s_mac

# Model and runtime configuration choices. A copy of this will be saved along with the model
# weights so that it is easy to reproduce later.

# Data
dataset = 'TDIUC'
data_path = f'/home/qzhb/dorren/VQA_Experiment/data/{dataset}'
img_feat = 'resnetpq_qtype'  # updn, resnet, updnmkii, resnetmkii
mkii = False  # If you want to also load codebook indices
data_subset = 1.0
d = Dictionary.load_from_file(f'data/dictionary_{dataset}.pkl')

map_path = f'{data_path}/map_{dataset}_features.json'

train_file = f'{data_path}/train_{dataset}.h5'
val_file = f'{data_path}/val_{dataset}.h5'
icarl_rehearsal_file = f'{data_path}/icarl_rehearsal_{dataset}.h5'

train_batch_size = 512
val_batch_size = 512
num_classes = 218  # 1480  # Number of classifier units 1480 for TDIUC, 31xx for VQA,28 for CLEVR

train_on = 'full'
test_on = 'full'  # 'full' or 'valid'

arrangement = dict()

only_first_k = dict()
only_first_k['train'] = sys.maxsize  # Use sys.maxsize to load all
only_first_k['val'] = sys.maxsize  # Use sys.maxsize to load all

qnorm = True  # Normalize ques feat?
imnorm = True  # Normalize img feat?

shuffle = False

fetch_all = False

if fetch_all:  # For ques_type, ans_class or ans_type arrangement, get all qualifying data
    assert (not shuffle)
    train_batch_size = 1
    val_batch_size = 1  # Dataset[i] will return all qualifying data of idx 1

load_in_memory = False
use_all = False
use_pooled = False
use_lstm = True

# Training
overwrite_expt_dir = True  # Set to True during dev phase
max_epochs = 40 
test_interval = 8

# Model
attn_type = 'old'  # new or old
num_attn_hops = 2
soft_targets = False
bidirectional = True
lstm_out = 512
emb_dim = 300
cnn_feat_size = 2048  # 2048 for resnet/updn/clevr_layer4 ; 1024 for clevr layer_3

classfier_dropout = True
embedding_dropout = True
attention_dropout = True
num_hidden = 1024
# use_model = vqa_models.UpDown  # BLAH
optimizer = torch.optim.Adamax
lr = 2e-3
save_models = False
if not soft_targets:
    train_on = 'valid'
num_rehearsal_samples = 50
buffer_replacement_strategy = 'random'
use_exponential_averaging = False
