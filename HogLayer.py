import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


def timeit(x, func, iter=10):
	torch.cuda.synchronize()
	start = time.time()
	for _ in range(iter):
		y = func(x)
	torch.cuda.synchronize()
	runtime = (time.time()-start)/iter
	return runtime

class HOGLayer(nn.Module):
    def __init__(self, nbins=10, pool=8, stride=1, padding=1, dilation=1):
        super(HOGLayer, self).__init__()
        self.nbins = nbins
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pool = pool
        mat = torch.FloatTensor([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        mat = torch.cat((mat[None], mat.t()[None]), dim=0)
        self.register_buffer("weight", mat[:,None,:,:])
        self.pooler = nn.AvgPool2d(pool, stride=pool, padding=0, ceil_mode=False, count_include_pad=True)

    def forward(self, x):
        with torch.no_grad():
            gxy = F.conv2d(x, self.weight, None, self.stride,
                            self.padding, self.dilation, 1)
            #2. Mag/ Phase
            mag = gxy.norm(dim=1)
            phase = torch.atan2(gxy[:,0,:,:], gxy[:,1,:,:])

            #3. Binning Mag with linear interpolation
            phase_int = (phase / math.pi * self.nbins)
            phase_int = phase_int[:,None,:,:]

            n, c, h, w = gxy.shape
            out = torch.zeros((n, self.nbins, h, w), dtype=torch.float, device=gxy.device)
            out.scatter_(1, phase_int.floor().long()%self.nbins, mag[:,None,:,:])
            out.scatter_add(1, phase_int.ceil().long()%self.nbins, 1-mag[:, None, :, :])

            return self.pooler(out)


if __name__ == '__main__':
    import cv2
    cuda = True

    path = '/home/etienneperot/Pictures/pencils.jpg'
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    x = torch.from_numpy(im)[None, None]
    if cuda:
        x = x.cuda().float()
    else:
        x = x.float()

    hog = HOGLayer(nbins=12, pool=2)
    if cuda:
        hog = hog.cuda()

    y = hog(x)

    y2 = y.cpu().numpy()
    bin = 0
    while 1:
        im = y2[0, bin]
        im = (im-im.min())/(im.max()-im.min())
        print('bin: ', bin, ' ang: ', float(bin)/hog.nbins * 180.0)
        cv2.imshow('bin', im)
        cv2.waitKey()
        bin = (bin + 1) % hog.nbins
