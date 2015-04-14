from data_reader import News20
from rsm import RSM
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import time
import numpy
import os

'''
Basic untested version of Replicated Softmax:
@inproceedings{hinton2009replicated,
  title={Replicated softmax: an undirected topic model},
  author={Hinton, Geoffrey E and Salakhutdinov, Ruslan R},
  booktitle={Advances in neural information processing systems},
  pages={1607--1614},
  year={2009}
}
based on Theano RBM implementation from deeplearning.net.
To test download the matlab preprocessed dataset from http://qwone.com/~jason/20Newsgroups/
and put it in the appropriate folder (see below).
This implementation is supposed to provide a framework for further work on RSM,
not just the good results right away.
@author: Michal Lisicki
'''

def compile_and_train_rsm(data_x,
	learning_rate=0.1,
	training_epochs=15,
	batch_size=20,
	n_chains=20, 
	n_hidden=20):
	
	train_set_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=True)
	n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
	if((float(train_set_x.get_value(borrow=True).shape[0]) / batch_size)!=float(n_train_batches)):
		n_train_batches+=1

	index = T.lscalar()
	x = T.matrix('x')

	rng = numpy.random.RandomState(123)
	theano_rng = RandomStreams(rng.randint(2 ** 30))

	persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden), dtype=theano.config.floatX), borrow=True)

	rsm = RSM(input=x, n_visible=train_set_x.get_value(borrow=True).shape[1],
			  n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

	cost, updates = rsm.get_cost_updates(lr=learning_rate, persistent=None, k=1)

	train_rsm = theano.function(
		[index],
		cost,
		updates=updates,
		givens={
			x: train_set_x[index * batch_size: (index + 1) * batch_size]
		},
		name='train_rbm'
	)

	start_time = time.time()

	for epoch in xrange(training_epochs):
		mean_cost = []
		for batch_index in xrange(n_train_batches):
			mean_cost += [train_rsm(batch_index)]

		print 'Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost)

	end_time = time.time()

	pretraining_time = (end_time - start_time)

	print ('Training took %f seconds' % (pretraining_time))
	return rsm

if __name__ == '__main__':
	# download dataset from http://qwone.com/~jason/20Newsgroups/
	train_dataset=News20(path='data/20news-bydate_preprocessed',which_set="train",vector_size=200)
	x,y = train_dataset.getDocumentWordFrequencies()
	x=x[:1000,:200] # limit number of documents and features for testing
	rbm=compile_and_train_rsm(x)
	
	#example of accessing the data
	print rbm.params[0].get_value() # weights
	print rbm.params[1].get_value() # hidden biases
	print rbm.params[2].get_value() # visible biases
	

 
