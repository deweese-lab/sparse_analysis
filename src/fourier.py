import numpy as np
import pickle
import scipy.stats as stats

class FourierTransform(object):
    """Class for computing fourier transforms and power spectra for a set of 2D images, passed as a list of 1D matrices. all images must be of same length and the lengths must be perfect squares."""
	def __init__(self, images):
        	self.images = images
        	self.dims = len(images[0])

	def compute_fourier_transform(self):
       		"""Function for computing fourier transform for each basis function."""
        	#get complex valued amplitudes of all Fourier components for each basis function
        	fourier_images = np.array([np.fft.fftn(self.images.reshape((self.dims,self.dims))) for self.dims in self.dims])
        	#square amplitudes to compute the variances (assuming mean amplitudes are 0 for each bf)
        	fourier_amplitudes = np.abs(fourier_images)**2
        	#reshape bfs into vectors
        	fourier_amplitudes = [amp.reshape(self.dims**2) for amp in fourier_amplitudes]
        	self.fourier_amplitudes = fourier_amplitudes

    	def construct_wave_vector_array(self):
        	"""Function for constructing array of wave vectors in k space"""
        	kfreq = np.fft.fftfreq(self.dims)*self.dims
        	kfreq2D = np.meshgrid(kfreq,kfreq)
        	knrm = np.sqrt(kreq2D[0]**2+kfreq2D[1]**2)
		return(knrm.flatten())

	def get_frequency_bins(self):
		kbins = np.arange(0.5, self.dims//2+1, 1.)
		return(kbins)
	
	def get_frequency_values(self, kbins):
		kvals = 0.5 * (kbins[1:] + kbins[:-1])
		return(kvals) 

    	def compute_power_spectrum(self):
        	wave_vectors = construct_wave_vector_array()
       		self.compute_fourier_transform()
        	kbins = get_frequency_bins()
        	kvals = get_frequency_values(kbins)
        	Abins = [stats.binned_statistic(wave_vectors, amplitude, statistic="mean", bins=kbins)[0] for amplitude in self.fourier_amplitudes]
        	Abins = [onebin*np.pi*(kbins[1:]**2 - kbins[:-1]**2) for onebin in Abins]
        	self.power_spectrum = (kvals, Abins)

	def get_max_frequencies(self):
		kvals = self.power_spectrum[0]
		Abins = self.power_spectrum[1]
		return np.array([kvals[np.argmax(Abin)] for Abin in Abins]


