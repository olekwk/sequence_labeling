import os
import numpy as np
import keras.backend as K


from custom_keras import Score
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Convolution1D,Masking, BatchNormalization,Input,merge,LSTM, Dropout,Embedding,Bidirectional
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.regularizers import l2
#import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import os
import pickle
from Bio.PDB.Polypeptide import standard_aa_names, aa1, d1_to_index
import pandas as pd


class Splitter(object):
	
	def __init__(self,label,valid_num):
		self.label = label
		self.valid_num = valid_num
	
	def split(self,sequences,save_path = None):
		
		def cnt(sequences):
			return sum([i.ss.count(self.label) for i in sequences])

		let = []
		data = np.array(sequences)
		#labels= np.array(labels)
		ind = range(len(data))
		ll = int((1/self.valid_num)*len(data))
		sp = []
		tru = cnt(data,label)
		for i in range(valid_num):
			prt = 0
			curr_error = abs(prt-tru)
			ct = 0
			while curr_error > 0.00001:
			  	
				part = np.random.choice(ind,ll,replace = False)
		
				wyb_d = data[part]
				if curr_error < pas:
					st_data = data[part]	        
				prt = cnt(wyb_d,'I')
				if ct == 10000:
					wyb_d = st_data[:]
					break
				ct += 1
			let.append(wyb_d)
			rem = [i for i in ind if i not in part]
			ind = rem[:]
		wyb_d = data[ind]
		let.append(wyb_d)

		if save_path:
			test = []
			for i in let[0]:
				for j in i:
					test.append(j)
			train = []
			for p in let[1:]:
				for i in p:
					for j in i:
						train.append(j)
			np.save(save_path+'test.npy',let[0])
			np.save(save_path+'train.npy',let[1:])
		return let

class Predictor(object):
		
	def __init__(self,model=None):
		
		self.weights = []
		self.model = model
		self.input_dim = ()
		
	def build_model(self,dim=(700,40),out = 4, learning_rate=0.0003,regularization_rate=0.0001):

		lr=learning_rate
		rl2 = regularization_rate
		inp1 = Input(shape=dim, dtype='float32', name='inp')
		a = Convolution1D(64,3,activation='tanh',padding='same',kernel_regularizer=l2(0.0001))(inp1)
		a_b = BatchNormalization()(a)
		b = Convolution1D(64,5,activation='tanh',padding='same',kernel_regularizer=l2(0.0001))(inp1)
		b_b=BatchNormalization()(b)
		e = Convolution1D(64,7,activation='tanh',padding='same',kernel_regularizer=l2(0.0001))(inp1)
		e_b =BatchNormalization()(e)
		x = merge([a_b,b_b,e_b], mode='concat', concat_axis=-1)
		t = TimeDistributed(Dense(200,activation='relu',kernel_regularizer=l2(0.0001)))(x)
		k = Bidirectional (LSTM(200, return_sequences=True, activation='tanh', recurrent_activation='sigmoid',dropout = 0.5,recurrent_dropout = 0.5))(t)
		k1 = Bidirectional (LSTM(200, return_sequences=True, activation='tanh', recurrent_activation='sigmoid',dropout = 0.5, recurrent_dropout = 0.5))(k)
		f=TimeDistributed(Dense(200,activation='relu',kernel_regularizer=l2(0.0001)))(k1)
		out = TimeDistributed(Dense(out,activation='softmax',name='out'))(f)
		model = Model(inputs=inp1, outputs=out)
		adam=Adam(lr=lr)
		model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
		self.model = model
		return model
			
		 	
	def fit(self,sequences = [],spliter = None,seq_list = [],epochs = 80,weights_path='' ):

			seq = splitter.split(sequences)[1:]
			self.test_set = seq[0]
			train_data =[]
			for i in seq:
				for j in i:
					train_data.append(j.representataion)								
		
			for i in range(nn):
					pom = np.concatenate(train_data[:i]+train_data[i+1:])
				 
					valid = train_data[i]
					ch1 = Score(train[:,:,data_id],train[:,:,label_id],lab_pos=2,path=weights_path+str(i)+'__')
					callbacks_list = [ch1]
					self.model.fit(train[:,:,data_id],train[:,:,label_id],epochs=epochs, batch_size=64, validation_data=(valid[:,:,data_id],valid[:,:,label_id]), callbacks=callbacks_list,verbose=2)

			weights = [(float(i.split('_')[0].split('__')[0]), float(i.split('_')[-2]),weights_path+i) for i in os.listdir(weights_path)]				
			weights.sort(key =lambda x: x[1],order='descending')
			sl = {}
			for i in weights:
				if i[0] not in sl:
					sl[i[0]] = i
			self.weights = list(sl.values)

	def load_weights(self,weights_path):
			weights = [(float(i.split('_')[0].split('__')[0]), float(i.split('_')[-2]),weights_path+i) for i in os.listdir(weights_path) if '.h5' in i]				
			weights.sort(key =lambda x: x[1],reverse = True)
			sl = {}
			for i in weights:
				if i[0] not in sl:
					sl[i[0]] = i
			self.weights = [i[-1] for i in sl.values()]
				
				
	def predict(self,custom_set = []):

		cor = self.model.input.shape[-1]
		
		if len(custom_set) and len(self.weights):
			self.test_set = custom_set
		pred_inp = []
		for j in self.test_set:
			for i in j.representation:
				pred_inp.append(i)
		pred_inp = np.asarray(pred_inp)
		pred_inp = np.concatenate((pred_inp,np.zeros((pred_inp.shape[0],pred_inp.shape[1],cor-pred_inp.shape[-1]))), axis=-1 )
		print(pred_inp.shape)
		#pred_inp = np.array([i for i in [j.representation for j in self.test_set]])
		result = np.zeros((pred_inp.shape[0],pred_inp.shape[1],self.model.output.shape[-1]))	
		for i in self.weights:
			self.model.load_weights(i)
			result += self.model.predict(pred_inp)
		result/=len(self.weights)
		sl = dict((j,i) for i,j in custom_set[0].lab_to_index.items())
		for i in self.test_set:
			sd = result[:i.representation.shape[0],:len(i.seq),:]
			i.prediction_proba = sd
			i.predicted_seq = ''.join([sl[i] for i in np.reshape(np.argmax(sd,axis=-1),(sd.shape[0]*sd.shape[1]))]) 


