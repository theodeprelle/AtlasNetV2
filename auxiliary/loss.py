from __future__  import print_function
import sys
import torch

sys.path.append("./extension/")
import dist_chamfer as ext
distChamferL2 =  ext.chamferDist()

def ChamferLoss(target,prediction):

    dist1, dist2 = distChamferL2(target, prediction)
    loss = torch.mean(dist1)+torch.mean(dist2)

    return loss

class LOSS_LIST:
    """list of all the model"""
    def __init__(self):

        self.losses={"AtlasNet":ChamferLoss,
                     "PatchDeformation":ChamferLoss,
                     "PointTranslation":ChamferLoss,}

    def load(self,options):

        loss = self.losses[options.model]
        return loss
