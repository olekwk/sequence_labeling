from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras import backend as K



from keras.models import Sequential
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import custom_object_scope
from keras import callbacks

from sklearn.metrics import roc_curve,auc,precision_score,recall_score,f1_score
from sklearn import preprocessing
import numpy as np

class Score(callbacks.Callback):
	def __init__(self, X_train, y_train, lab_pos=2,useCv = True,path=''):
		super(Score, self).__init__()
		self.bestAucCv = 0
		self.bestAucTrain = 0
		self.cvLosses = []
		self.bestCvLoss = 1,
		self.X_train = X_train
		self.y_train = y_train
		self.useCv = useCv
		self.lab_pos = lab_pos
		self.f1 = 0
		self.path = path
	def on_train_begin(self, logs={}):
		return

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):
		
		train_pred = self.model.predict(np.array(self.X_train))
		#wyb = np.argmax(train_pred,axis=-1)
		scr_tr,wyb_tr,tr_tr=[],[],[]
		for j,j1 in zip(self.y_train,train_pred):
			for i,i1 in zip(j,j1):
				if max(i) == 1:
					tr_tr.append(np.argmax(i))
					wyb_tr.append(np.argmax(i1))
					scr_tr.append(i1[self.lab_pos])
		
		rec_tr = recall_score(tr_tr,wyb_tr,average='micro',labels=[self.lab_pos])
		prec_tr = precision_score(tr_tr,wyb_tr,average='micro',labels=[self.lab_pos])
		f1_tr=f1_score(tr_tr,wyb_tr,average='micro',labels=[self.lab_pos])
		fpr, tpr, thresholds = roc_curve(tr_tr, scr_tr, pos_label=self.lab_pos)
		aucTrain = auc(fpr,tpr)
		print("SKLearn Train AUC score: " + str(aucTrain))
	
		if (self.bestAucTrain < aucTrain):
			self.bestAucTrain = aucTrain
			print ("Best SKlearn AUC training score so far")
			#**TODO: Add your own logging/saving/record keeping code here

		if (self.useCv) :
			cv_pred = self.model.predict(self.validation_data[0])
			scr_val,wyb_val,tr_val=[],[],[]	
			for j,j1 in zip(self.validation_data[1],cv_pred):
				for i,i1 in zip(j,j1):
					if max(i) == 1:
						tr_val.append(np.argmax(i))
						wyb_val.append(np.argmax(i1))
						scr_val.append(i1[self.lab_pos])
			rec_val = recall_score(tr_val,wyb_val,average='micro',labels=[self.lab_pos])
			prec_val = precision_score(tr_val,wyb_val,average='micro',labels=[self.lab_pos])
			f1_val=f1_score(tr_val,wyb_val,average='micro',labels=[self.lab_pos])
			
			fpr, tpr, thresholds = roc_curve(tr_val, scr_val, pos_label=self.lab_pos)
			aucCv = auc(fpr,tpr)
			print(str(epoch))
			print ("Val AUC score: " +  str(aucCv))
			print('Train F1_score '+str(f1_tr)+' '+'Train sens: '+str(rec_tr)+' Train prec: '+str(prec_tr))
			print('Val F1_score '+str(f1_val)+' '+'Val sens: '+str(rec_val)+' Val prec: '+str(prec_val))
			
	
			if (self.f1 < f1_val):
				# Great! New best *actual* CV AUC found (as opposed to the proxy AUC surface we are descending)
				print("Best f1 score")
				self.f1 = f1_val

				# **TODO: Add your own logging/model saving/record keeping code here.
				self.model.save(self.path+"best_auc_model" + str(epoch)+'__'+str(f1_tr)+'_'+str(rec_val)+'_'+str(prec_val)+'_'+str(f1_val)+'_'+str(aucCv)+".h5", overwrite=True)

			vl = logs.get('val_loss')
			if (self.bestCvLoss < vl) :
				print("Best val loss on SoftAUC so far: "+str(self.bestCvLoss) )
				#**TODO -  Add your own logging/saving/record keeping code here.
		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		# logs include loss, and optionally acc( if accuracy monitoring is enabled).
		return


