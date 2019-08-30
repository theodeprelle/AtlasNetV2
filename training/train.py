from __future__ import print_function
import argparse
import sys
import torch
import torch.utils.data
import datetime
from torch.autograd import Variable
import matplotlib.cm as cm

sys.path.insert(1,'.')
from auxiliary.loss import *
from auxiliary.model import *
from auxiliary.utils import *
from auxiliary.dataset import *

#========================================================================================
#                                   argument parsing
#========================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--patchDim', type=int, default = 2,  help='')
parser.add_argument('--patchDeformDim', type=int, default = 3,  help='')
parser.add_argument('--model', type=str, default ="AtlasNet", help='')
parser.add_argument('--adjust', type=str, default ="mlp", help='')
parser.add_argument('--lrate', type=float, default =0.001, help='')
parser.add_argument('--nbatch', type=int, default = 16, help='')
parser.add_argument('--nepoch', type=int, default = 325, help='')
parser.add_argument('--npoint', type=int, default = 2500,  help='')
parser.add_argument('--npatch', type=int, default = 10,  help='')
parser.add_argument('--dataset', type=str, default ="shapenet", help='')
parser.add_argument('--nlatent', type=int, default =1024, help='')
parser.add_argument('--loadmodel', type=str, default=None,help='')
parser.add_argument('--firstdecay', type=int, default = 250,  help='')
parser.add_argument('--seconddecay', type=int, default = 300,  help='')
parser.add_argument('--training_id', type=str, default=None,help='')
opt = parser.parse_args()

if opt.training_id == None and opt.model in ['PointTranslation','AtlasNet']:

    opt.training_id = "%s%dD_%sAdj_%s_%dpatch_%dpts_%dep"%(opt.model,
                			                               opt.patchDim,
                                                           opt.adjust,
                                                           opt.dataset,
                                                           opt.npatch,
                                                           opt.npoint,
                                                           opt.nepoch)

if opt.training_id == None and opt.model == 'PatchDeformation':

    opt.training_id = "%s%dDto%dD_%s_%s_%dpatch_%dpts_%dep"%(opt.model,
                			                                 opt.patchDim,
                                                             opt.patchDeformDim,
                                                             opt.adjust,
                                                             opt.dataset,
                                                             opt.npatch,
                                                             opt.npoint,
                                                             opt.nepoch)
display_opts(opt)
#========================================================================================


#========================================================================================
#                                   training element
#========================================================================================
DATASET = DATASET_LIST()                                       #list of all the methods
MODEL = MODEL_LIST(opt)                                        #list of all the models
LOSS = LOSS_LIST()                                             #list if all the losses
#========================================================================================


#========================================================================================
#                                    model selection
#========================================================================================
if opt.model not in MODEL.type:
    print(COLORS.FAIL,"ERROR please select the model from : ",COLORS.ENDC)
    for model in sorted(MODEL.type):
        print("   >",model)
    exit()

network = MODEL.load(opt)                                      # load the loss
network.apply(weights_init)                                    # init the weight

if opt.loadmodel != None:
    network.load_state_dict(opt.loadmodel)
#========================================================================================


#========================================================================================
#                                    dataset selection
#========================================================================================
if opt.dataset not in DATASET.type:
    print(COLORS.FAIL,"ERROR please select the dataset from : ",COLORS.ENDC)
    for dataset in DATASET.type:
        print("   >",dataset)
    exit()

dataset_train = DATASET.load(training=True,options=opt)
dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                               shuffle=True,
                                               batch_size=opt.nbatch,
                                               num_workers=12)

dataset_valid = DATASET.load(training=False,options=opt)
dataloader_valid = torch.utils.data.DataLoader(dataset_valid,
                                               shuffle=False,
                                               batch_size=opt.nbatch,
                                               num_workers=12)
#======================================================================================


#======================================================================================
#                                 loss selection
#======================================================================================
loss = LOSS.load(opt)
#======================================================================================


#======================================================================================
#                                 optimizer
#======================================================================================
optimizer = torch.optim.Adam(network.parameters(),lr = opt.lrate)
#======================================================================================


#======================================================================================
#                               training logs
#======================================================================================
logger_path = "./log/%s"%opt.training_id

if not os.path.exists('log'):
    os.mkdir('log')

if not os.path.exists(logger_path):
    os.mkdir(logger_path)

with open('%s/opt.pickle'%logger_path, 'wb') as handle:
    pickle.dump(opt, handle, protocol=pickle.HIGHEST_PROTOCOL)

trainloss_log = LOGGER()
validloss_log = LOGGER()
visdom = visdom.Visdom(env=opt.training_id, port=8888)
#======================================================================================


#======================================================================================
#                                   TRAINING BEGIN
#======================================================================================
for epoch_id in range(opt.nepoch):

    network.train()

    trainloss_log.reset()

    if epoch_id == opt.firstdecay:
        optimizer = torch.optim.Adam(network.parameters(),lr = opt.lrate/10)

    if epoch_id == opt.seconddecay:
        optimizer = torch.optim.Adam(network.parameters(),lr = opt.lrate/100)


    #==================================================================================
    #                                   training
    #==================================================================================
    for batch_id, batch in enumerate(dataloader_train):

        if batch.size(0) == opt.nbatch:

            optimizer.zero_grad()

            batch = batch.cuda()
            prediction, learnedPatches = network(batch.cuda())
            fittingLoss = loss(batch,prediction)

            fittingLoss.backward()
            optimizer.step()

            trainloss_log.add(fittingLoss.item())
            display_it("train", opt, epoch_id, batch_id, trainloss_log.mean())
    #==================================================================================


    #==================================================================================
    #                                validation
    #==================================================================================
    with torch.no_grad():

        network.eval()

        for batch_id, batch in enumerate(dataloader_valid):

            batch = batch.cuda()
            prediction, learnedPatches = network(batch.cuda())
            fittingLoss = loss(batch,prediction)

            validloss_log.add(fittingLoss.item())
            display_it("valid", opt, epoch_id, batch_id, validloss_log.mean())
    #==================================================================================

    #==================================================================================
    #                          saving loss and parameters
    #==================================================================================
    if len(validloss_log.history) > 0:
        X = np.column_stack((train_log.history, valid_log.history))
        Y = np.column_stack((np.arange(len(train_log.history)),
                             np.arange(len(train_log.history))))
        visdom.line(X, Y,  win="Fitting loss",
                    opts=dict(title="Fitting loss", legend=["train", "valid"]))

    color =  [[125,125,125]]*(batch.size(1))
    cmap = cm.get_cmap('hsv')
    for i in range(opt.npatch):
        c = cmap(i/opt.npatch-1)
        color += [[int(c[0]*255),int(c[1]*255),int(c[2]*255)]]*(opt.npoint//opt.npatch)

    color = np.array(color)
    if batch_id < 3:
        X = np.vstack((batch[0].data.cpu().numpy(),prediction[0].data.cpu().numpy()))
        Y = np.array([1]*batch.size(1)+[2]*opt.npoint)
        visdom.scatter(X,
                       Y,
                       win='gt%d' % batch_id,
                       opts=dict(markersize=4,
                                 markercolor=color,
                                 title='gt%d' % batch_id,
                                 legend=['gt', 'batch']))

    torch.save(network.state_dict(), '%s/network.pth' % (logger_path))
    #==================================================================================i
