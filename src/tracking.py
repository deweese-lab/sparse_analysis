import numpy as np
import pickle 
import scipy.stats as stats
import fourier as fourier
#import plotting (utils) as plotting

class TrainedDictionary(object):
	"""Class for classifying basis functions into groups and analyzing histories of sparse coding dictionaries."""
	def __init__(self, bfhistory, fourier_transform=None):
        	self.bfhistory = bfhistory
        	self.get_similarity_over_time(-1)
        	self.get_fourier_transforms(fourier_transform)		
    
    	def get_fourier_transforms(self, fourier_transform):
        	if fourier_transform:
            		pass
        	else:
			fourier_transform.compute_power_spectrum()
			
	def compute_similarity(self, bfs, referencebfs):
		"""Function for computing cosine similarity between a batch of basis functions and a corresponding batch of reference basis functions. Must pass batches with batch-major axes."""
        	return np.einsum('ij,ij->i', bfs, referencebfs)/np.linalg.norm(referencebfs, axis=1)/np.linalg.norm(bfs, axis=1)
        
    	def get_similarity_over_time(self, ref_idx):
        	similarities_over_time = []
        	for bfs in self.bfhistory:
            		similarities_over_time.append(self.compute_similarity(bfs, self.bfhistory[ref_idx]))
        	self.similarities_over_time = similarities_over_time

	def split_bfs_into_max_freq_groups(nsplits=None):
		max_freqs = fourier_transform.get_max_frequencies() 
		
		if nsplits:
			groups = np.split(np.argsort(max_freqs[:(len(fourier_transform.images)-(len(fourier_transform.images)%nsplits))],nsplits))
		
		else:
			groups = []
			num_cycles = 1
			cycle_group = np.where(max_freqs == num_cycles)[0]
			while len(cycle_group) != 0: 
				groups.append(cycle_group)
				num_cycles += 1
				cycle_group = np.where(max_freqs == num_cycles)[0]
		return(groups)

    	def get_mean_timecourse(timecourses, timebounds):
        	mean_timecourse = np.array([timepoint.mean() for timepoint in timecourses][timebounds[0]:timebounds[1]])
        	std_err = np.array(np.array([1.96*rf.std()/np.sqrt(np.shape(timecourses)[1]) for timepoint in timecourses][timebounds[0]:timebounds[1]])
        	return(mean_timecourse, std_err)

    	def get_similarity_over_time_by_group_mean(self, groups, timebounds):
		means = []
		stdevs = []
		for group in groups:
			grouphistory = [timepoint[group] for timepoint in self.similarities_over_time]
			mean, stdev = get_mean_timecourse(grouphistory, timebounds)
			means.append(mean)
			stdevs.append(stdev)
		return means, stdevs
	    
    	def plot_timecourses_by_group_mean(self, split_timepoint, labels, nsplits=None, metric='similarity', timebounds=(0,100), xbounds=[-2,102], xtickmarks=[0,50], ybounds=[0,1.02], ytickmarks=[0,1]): 
		groups =  split_bfs_into_max_freq_groups(nsplits=nsplits)	
		means, stdevs = get_similarity_over_time_by_group_mean(groups, timebounds)
		
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)

		font = {'fontname':'Arial'}
		
		for mean, stdev, label in zip(means, stdevs, labels):
			x = np.linspace(0, len(mean), len(mean))
			ax.fill_between(x, mean-stdev, mean+stdev, label=label, alpha=0.5)
		
		ax.set_xlabel('Training iterations ($t$)', **font)
		ax.set_ylabel(r"$\bf{" + similarity + "}$" + '$(BF_t, BF_T)$', **font)
		
		ax.set_xlim(xbounds)
		ax.set_ylim(ybounds)

		ax.set_xticks(xtickmarks, fontsize=36, **font)
		ax.set_yticks(ytickmarks, fontsize=36, **font)
			
		return(fig,ax)


    #put fourier transform into different module and import it into the tracking module
    #add instance of fourier transform class into a function in traineddictionary
    #convert fourier transform object to having all bound variables and pass bound variables into trained dictionary bound variables
    #add get max frequency to bound functions


        
        
