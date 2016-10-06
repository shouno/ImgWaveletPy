# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import scipy.misc
from ImgWavelet import rgb2gray, ImageWavelet
import matplotlib.pylab as plt


##
## 画像を 128 x 128 のブロックに分解して，
## それぞれのブロックのウェーブレット表現を作ってみる
##

nH = 128
nV = 128
stride = 128

#l = scipy.misc.lena()
#l = np.double(l) / 256.
#l = (l - l.mean()) / l.std()
#

l = scipy.misc.face()  # これだと RGB カラーのたぬきなので，512x512 に切り出しのうえ，平均０，分散１の絵にしておく
l = rgb2gray( l[200:(200+512),300:(300+512)] )
l = np.double(l) / 256.
l = (l - l.mean()) / l.std()


cdat = []
aa = ImageWavelet( np.zeros( (nV, nH) ) ) # 零埋め画像のImageWaveletオブジェクト
for ytop in range( 0, l.shape[0], stride ):
    for xlft in range( 0, l.shape[1], stride ):
        ll = l[ytop:ytop+nV, xlft:xlft+nH]
        cdat.append( aa.Wv2coeff( ll ) )

cdat = np.array( cdat )


#
# これで cdat に 16 ブロック× 128x128 の画像ができるはず．
#


plt.figure()
for i in range(16):
    plt.subplot( 4, 4, i+1 )
    img = aa.Coeff2Wv( cdat[i,:] )
    plt.imshow( img, cmap='gray', interpolation='nearest' )
    plt.axis('off')
plt.suptitle( 'Reconstructed image with Haar bases' )
plt.show()

plt.figure()
for i in range(16):
    plt.subplot( 4, 4, i+1 )
    plt.plot( cdat[i,:] )
    plt.xlim( (0,nH*nV) )
    plt.xticks(np.array([0,1,2,3,4])*4096)
    plt.grid()
plt.suptitle( 'Wavelet Coeffs for each block' )
plt.show()
