# -*- coding: utf-8 -*-
#
#

"""
Image Wavelet クラスは，2の冪サイズのイメージを与えて
そのウェーブレット表現を扱うためのクラス
"""

import numpy as np
import pywt

def rgb2gray(rgbimg):
    """Change RGB image to gray one"""
    return np.dot(rgbimg[...,:3], [0.299, 0.587, 0.114])


class ImageWavelet:
    def __init__( self, imgmat, mode='db1' ):
        self.img = imgmat
        self.mode = mode
        self.coeffs = pywt.wavedec2( self.img, self.mode )
        self.maxlvl = len( self.coeffs )
        self.nH = 2**(self.maxlvl-1)
        self.nV = 2**(self.maxlvl-1)

    def BaseMat( self ):
        coeffs = pywt.wavedec2( np.zeros( (self.nV, self.nH) ), self.mode )
        nl = self.nV*self.nH
        Phi = np.matrix( np.zeros( (nl, nl) ) )
        coeffs[0][0] = 1.
        img = pywt.waverec2( coeffs, mode )
        Phi[:,0] = img.reshape( (nl,1) )
        coeffs[0][0] = 0.
        clm = 1

        for lv in range( 1, self.maxlvl ):
            cH, cV, cD = coeffs[lv]
                         
            shp = cH.shape
            cnum = np.prod( shp )
            for i in range( cnum ):
                q = np.zeros( shp )
                q.reshape( (cnum,) )[i] = 1.
                coeffs[lv] = (q, cV, cD)
                img = pywt.waverec2( coeffs, self.mode ).reshape( (nl,1) )
                Phi[:,clm] = img
                clm = clm + 1
                coeffs[lv] = ( cH, cV, cD )

            shp = cV.shape
            cnum = np.prod( shp )
            for i in range( cnum ):
                q = np.zeros( shp )
                q.reshape( (cnum,) )[i] = 1.
                coeffs[lv] = (cH, q, cD)
                img = pywt.waverec2( coeffs, self.mode ).reshape( (nl,1) )
                Phi[:,clm] = img
                clm = clm + 1
                coeffs[lv] = ( cH, cV, cD )

            shp = cD.shape
            cnum = np.prod( shp )
            for i in range( cnum ):
                q = np.zeros( shp )
                q.reshape( (cnum,) )[i] = 1.
                coeffs[lv] = (cH, cV, q)
                img = pywt.waverec2( coeffs, self.mode ).reshape( (nl,1) )
                Phi[:,clm] = img
                clm = clm + 1
                coeffs[lv] = ( cH, cV, cD )

    def Wv2coeff( self, img=None ):
        """
        Return array means coefficients for the Wavelet bases 
        expressed as column vectors
        """
        if img is None:
            img = self.img
        else:
            if img.shape != self.img.shape:
                raise Exception( 'Image Sizes incompatible' )
        coeffs = pywt.wavedec2( img, self.mode )
        nl = self.nH * self.nV
        ret = np.zeros( nl )
        ret[0] = coeffs[0][0][0]
        idx = 1
        for lv in range( 1, self.maxlvl ):
            for cc in coeffs[lv]:  # (cH, cV, cD)
                crow, cclm = cc.shape
                ret[idx:idx+crow*cclm] = cc.reshape( (crow*cclm,) )
                idx = idx + crow*cclm
        return ret

    def Coeff2Wv( self, coeff ):
        """
        Interprete coeffs as corresponding wavelet structure 
        and return image for the coeffs
        """
        nl = self.nH * self.nV
        assert len( coeff ) == nl
        self.coeffs[0][0][0] = coeff[0]
        idx = 1
        for lv in range( 1, self.maxlvl ):
            cH, cV, cD = self.coeffs[lv]

            crow, cclm = cH.shape
            clen = crow*cclm
            cHnew = coeff[idx:idx+clen].reshape( (crow, cclm) )
            idx = idx + clen

            crow, cclm = cV.shape
            clen = crow*cclm
            cVnew = coeff[idx:idx+clen].reshape( (crow, cclm) )
            idx = idx + clen

            crow, cclm = cD.shape
            clen = crow*cclm
            cDnew = coeff[idx:idx+clen].reshape( (crow, cclm) )
            idx = idx + clen

            self.coeffs[lv] = (cHnew, cVnew, cDnew)

        self.img = pywt.waverec2( self.coeffs, self.mode )
        return self.img
