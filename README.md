ImageWaveletPy
================================

Image Wavelete representation example for python
A wrapper for pyWavelet for simple image use.

If you'd like to use this wrapper, you should install PyWavelets
from pip, anaconda and so on.

The PyWavelets is a good implementation package for wavelet, however, 
many researchers might want only decomposed result of wavelet representation.

If you'd like use the wrapper (see the last part of the WvTest01.py)

    
	from ImgWavelet import ImageWavelet, rgb2gray
	import scipy.misc
	
	l = scipy.misc.face()
	l = rgb2gray( l[200:(200+512),300:(300+512)] )
	l = np.double(l) / 256.
	l = (l - l.mean()) / l.std()

	a = ImageWavelet( l )
	coefs = a.Wv2coeff()

This code generate a Wavelet coefficients vector `coefs` for image `l`. 
You can modify coefficients value in the `coefs` and decode it by the following code.

	img = a.Coeff2Wv( coefs )

Then you can obtain the corresponding image(see WvTest02.py).

