import torch
import dist_chamfer
dist =  dist_chamfer.chamferDist()

with torch.enable_grad():
   p1 = torch.rand(10,1000,6)
   p2 = torch.rand(10,1500,6)
   p1.requires_grad = True
   p2.requires_grad = True
   points1 = p1.cuda()
   points2 = p2.cuda()
   cost, _ = dist(points1, points2)
   print(cost)
   loss = torch.sum(cost)
   print(loss)
   loss.backward()
   print(points1.grad, points2.grad)
