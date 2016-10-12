# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import scipy.misc
from ImgWavelet import rgb2gray, ImageWavelet
import matplotlib.pylab as plt
from sklearn import linear_model

##
## 手順
## 1. 画像にノイズのせる
## 2. 画像を 32 x 32 のブロックに分解する
## 3. LASSO 回帰で各ブロックの係数を推定
## 4. 結果評価



## 手順1. 画像にノイズのせる

nH = 32
nV = 32
stride = 32

imgsize = 512

l = scipy.misc.face()  # これだと RGB カラーのたぬきなので，512x512 に切り出しのうえ，平均０，分散１の絵にしておく
l = rgb2gray( l[200:(200+imgsize),300:(300+imgsize)] )
l = np.double(l) / 256.
l = (l - l.mean()) / l.std()

sgmlvl = 0.4

ltrue = l  # 正解画像
lobs = l + np.random.randn( imgsize, imgsize ) * sgmlvl


## 手順2. 画像をブロック分割して回帰データを作っておく
yreg = []
cdat = []
aa = ImageWavelet( np.zeros( (nV, nH) ) ) # 零埋め画像のImageWaveletオブジェクト
X = aa.BaseMat() # yreg には基底がはいる．

for ytop in range( 0, lobs.shape[0], stride ):
    for xlft in range( 0, lobs.shape[1], stride ):
        ll = lobs[ytop:ytop+nV, xlft:xlft+nH]
        yreg.append( ll.reshape((nH*nV,)) - ll.mean() )  # 平均は引いておく
        cdat.append( aa.Wv2coeff( ll ) )
yreg = np.array( yreg )
cdat = np.array( cdat )

## 手順3. LASSO 回帰してみる
reg = linear_model.Lasso( alpha=5e-4, fit_intercept=False, tol=1e-6 )
reg.fit( X[:,1:], yreg.T )  # 平均部分は推定しない 0 にしているので

## 手順4. 結果評価
lrec = np.zeros( (imgsize, imgsize) )
cnt = 0
recdat = np.hstack( (cdat[:,0].reshape(reg.coef_.shape[0],1), reg.coef_) )
for ytop in range( 0, lrec.shape[0], stride ):
    for xlft in range( 0, lrec.shape[1], stride ):
        lrec[ytop:ytop+nV, xlft:xlft+nH] = aa.Coeff2Wv( recdat[cnt,:] )
        cnt += 1


vmin = np.min( (lobs, lrec, ltrue) )
vmax = np.max( (lobs, lrec, ltrue) )

plt.figure()
plt.subplot( 1, 3, 1 )
plt.imshow( l, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax )
plt.axis('off')
plt.title( 'True Image' )
plt.subplot( 1, 3, 2 )
plt.axis( 'off' )
plt.title( 'Reconstruct Image' )
plt.imshow( lrec, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax )
plt.subplot( 1, 3, 3 )
plt.axis( 'off' )
plt.title( 'Observed Image' )
plt.imshow( lobs, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax )
plt.show()




