import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/CommonGen/",
	nbatches = 500,
	threads = 8,
	sampling_mode = "normal",
	bern_flag = 0,
	filter_flag = 1,
	neg_ent = 64,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/CommonGen/", "link")

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 1024,
	p_norm = 1,
	norm_flag = True)

# define the loss function
model = NegativeSampling(
	model = transe,
	loss = MarginLoss(margin = 0.0),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 1, use_gpu = True)
trainer.run()
transe.save_checkpoint('./checkpoint/CommonGen_transe.ckpt')

# test the model
transe.load_checkpoint('./checkpoint/CommonGen_transe.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)

import pickle
import numpy as np
ent_embeddings = transe.ent_embeddings.weight.data.cpu().detach().numpy()
rel_embeddings = transe.rel_embeddings.weight.data.cpu().detach().numpy()
pickle.dump(ent_embeddings, open("./benchmarks/CommonGen/ent_embeddings","wb"))
pickle.dump(rel_embeddings, open("./benchmarks/CommonGen/rel_embeddings","wb"))