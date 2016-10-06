# -*- coding: utf-8 -*-

# ウェーブレットのテスト

import numpy as np
import pywt
import scipy.misc
import matplotlib.pylab as plt
from ImgWavelet import rgb2gray, ImageWavelet

#coeffs = pywt.wavedec2( np.ones( (8,8) ), 'db1', level=2 )

coeffs = pywt.wavedec2( np.zeros( (8,8) ), 'db1' )

# なるほど係数第0成分は DC で，1x1 の行列になるわけか
#coeffs[0][0,0] = 1
#img = pywt.waverec2( coeffs, 'db1' )

# 次 Lv1 1x1 の行列
# 第0成分は水平エッジ
#coeffs[1][0][0,0] = 1
#img = pywt.waverec2( coeffs, 'db1' )
#
# 第1成分は垂直
#coeffs[1][1][0,0] = 1
#img = pywt.waverec2( coeffs, 'db1' )

# 第2成分はななめ
#coeffs[1][2][0,0] = 1
#img = pywt.waverec2( coeffs, 'db1' )


# とりま Lv2 まで見ておくか
# Lv2 は，2x2 の行列
# 左斜め上の水平エッジ成分ね
# トポロジカルに並んでいるというわけですな．りょうかい
coeffs[2][0][0,0] = 1.
img = pywt.waverec2( coeffs, 'db1' )

coeffs[2][0][1,1] = 1.
img = pywt.waverec2( coeffs, 'db1' )


# というわけで，基底行列を作ってみよう．

lv = 2
nH = 2**lv
nV = 2**lv
coeffs = pywt.wavedec2( np.zeros( (nV,nH) ), 'db1' )
Phi = np.matrix( np.zeros((nH*nV, nH*nV)) )

# coeffs[0][0] だけ特別扱いしておく
coeffs[0][0] = 1.
img = pywt.waverec2( coeffs, 'db1' )
Phi[:,0] = img.reshape( (nH*nV,1) )
coeffs[0][0] = 0.
clm = 1


for lv in range( 1, lv+1 ):
    cH, cV, cD = coeffs[lv]
    
    shp = cH.shape
    cnum = np.prod( shp )
    for i in range( cnum ):
        q = np.zeros( shp )
        q.reshape( (cnum,) )[i] = 1.
        coeffs[lv] = (q, cV, cD)
        img = pywt.waverec2( coeffs, 'db1' ).reshape( (nV*nH,1) )
        Phi[:,clm] = img
        clm = clm + 1
        coeffs[lv] = (cH, cV, cD )

    shp = cV.shape
    cnum = np.prod( shp )
    for i in range( cnum ):
        q = np.zeros( shp )
        q.reshape( (cnum,) )[i] = 1.
        coeffs[lv] = (cH, q, cD)
        img = pywt.waverec2( coeffs, 'db1' ).reshape( (nV*nH,1) )
        Phi[:,clm] = img
        clm = clm + 1
        coeffs[lv] = (cH, cV, cD )

    shp = cD.shape
    q = np.zeros( shp )
    cnum = np.prod( shp )
    for i in range( cnum ):
        q = np.zeros( shp )
        q.reshape( (cnum,) )[i] = 1.
        coeffs[lv] = (cH, cV, q)
        img = pywt.waverec2( coeffs, 'db1' ).reshape( (nV*nH,1) )
        Phi[:,clm] = img
        clm = clm + 1
        coeffs[lv] = (cH, cV, cD )


# これで行列 Phi に基底ができているはず．

for r in range(4):
    for c in range(4):
        idx = r*4+c
        plt.subplot( 4, 4, idx+1 )
        plt.imshow( Phi[:,idx].reshape(4,4), cmap='gray', interpolation='nearest' )
        plt.axis('off')

plt.show()


#l = scipy.misc.lena()

l = scipy.misc.face()  # これだと RGB カラーのたぬきなので，512x512 に切り出しのうえ，平均０，分散１の絵にしておく
l = rgb2gray( l[200:(200+512),300:(300+512)] )
l = np.double(l) / 256.
l = (l - l.mean()) / l.std()

a = ImageWavelet( l )

plt.figure()
plt.subplot( 1, 2, 1 )
plt.imshow( l, cmap='gray', interpolation='nearest' )
plt.axis('off')
plt.subplot( 1, 2, 2 )
coefs = a.Wv2coeff()
plt.plot( coefs )
plt.xlim( (0, coefs.size) )
plt.grid()
plt.title( 'Wavelet Coeffs' )
plt.show()
