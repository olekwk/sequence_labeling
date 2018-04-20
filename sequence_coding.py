import os
import pickle
from Bio.PDB.Polypeptide import standard_aa_names, aa1, d1_to_index
import pandas as pd
from Bio import SeqIO
import numpy as np



class Encoder (object):

	def __init__(self,partition=700):
		self.partition=partition

	def set_transform_func(self):
		pass

	def create(self,seq_list,pssm=[],path = None):
		seq_list = [Sequence(i[0],ss=i[1]) if (type(i) == list or type(i)==tuple) and len(i)==2 and len(i[1]) else Sequence(i) for i in seq_list ]
		#return seq_list
		k=0
		if len(pssm):
			for i in seq_list:
				i.encode(partition=self.partition,pssm=path+pssm[k]+'.pssm')
				k+=1
		else:
			for i in seq_list:
				i.encode(partition=self.partition)
		return seq_list

			 





class Sequence(object):
	pssm_alph = 'ARNDCQEGHILKMFPSTWYV'
	d1_to_index = d1_to_index
	lab_to_index = {'E':0,'H':1,'I':2,'C':3}
	ordered_input = False
	ordered_output = False
	@classmethod
	def set_class_property(cls,input_alphabet=aa1,output_alphabet='EHIC',ordered_input = False,ordered_output=True):
			
		diff_letter =[]
		for i in input_alphabet:
			if i not in diff_letter:
				diff_letter.append(i)
		if not ordered_input:
			diff_letter.sort()
		if not ordered_output:
			output=[i for i in output_alphabet].sort()
		else:
			output=[i for i in output_alphabet]
		d1_to_index = dict((i,k) for i,k in enumerate(diff_letter))
		lab_to_index = dict((i,k) for i,k in enumerate(output))
		d1_to_index = dict((i,j) for j,i in d1_to_index.items())
		lab_to_index = dict((i,j) for j,i in lab_to_index.items())
		
		cls.d1_to_index = d1_to_index
		cls.lab_to_index = lab_to_index
		cls.ordered_input = ordered_input
		cls.ordered_output = ordered_output
	
	def __init__(self,seq,ss = None):
		self.seq = seq
		self.st=len(Sequence.lab_to_index)
		self.ss = ss
		self.predicted_proba = None
		self.predicted_seq = None

	def encode(self,pssm = None,sti=22,pssm_mode=1,partition = 700):
		pt = len(Sequence.d1_to_index)
		def code():
			#lab_to_index={'B':0,'C':1,'E':2,'G':3,'H':4,'I':5,'S':6,'T':7}
			z=[]
			
			if self.ss and len(self.ss)==len(self.seq):
				for i ,j in zip(self.seq,self.ss):
					s=[0]*pt
					s1=[0]*self.st
					s[Sequence.d1_to_index[i]]=1
				
					s1[Sequence.lab_to_index[j]]=1

					z.append(s+s1)
			else:
				for i in self.seq:
					s=[0]*pt
					s[Sequence.d1_to_index[i]]=1
					z.append(s)

			
			z=np.asarray(z)
			if self.ss and len(self.ss)==len(self.seq):
				z=np.concatenate((z,np.zeros(((partition-len(self.seq)%partition)%partition,pt+self.st))),axis=0)
			else:
				z=np.concatenate((z,np.zeros(((partition-len(self.seq)%partition)%partition,pt))),axis=0)

			zd=[]
			for i in range(0,len(z),partition):
				zd.append(z[i:i+partition])
			zd=np.asarray(zd)
			return zd
		if pssm:
			pss = open(pssm, 'r')
			k=0
			cv=[]
			dk = []
			if pssm_mode==1: 
				for i in pss:
					c=[j for j in i.strip().split(' ') if j]
					#print(c)
					if k>2 and len(c) > pt:
						c1=[1/(1+np.exp(-int(j))) for j in c[sti-pt:sti]]
						dw = [int(j) for j in c[sti-pt:sti]]
						cv.append(c1)
						dk.append(dw)
					k+=1
			else:
				for i in pss:
					c=[j for j in i.strip().split(' ') if j]
					#print(c)
					if k>2 and len(c) > pt:
						#c1=[1/(1+np.exp(-int(j))) for j in c[sti-20:sti]]
						dw = [int(j)/100.0 for j in c[sti:sti+pt]]
						cv.append(dw)
						dk.append(dw)
					k+=1

			dk = np.asarray(dk)
			cv=np.asarray(cv)
			cv=np.concatenate((cv,np.zeros(((partition-len(self.seq)%partition)%partition,pt))),axis=0)
			h = code()[1]
			#print(h.shape,cv.shape)		
			fetr = np.concatenate((h,cv),axis=-1)
			zd=[]
			for i in range(0,len(h),partition):
				zd.append(fetr[i:i+partition])
			zd=np.asarray(zd)
			self.pssm = dk
			#print(zd.shape)
			return zd
		
			
		else:
			self.pssm = None
			dp=code()
			self.representation = dp
  

	def highlight (self):
		if self.ss and len(self.ss) == len(self.seq):
			st=''
			for i ,j in zip(self.seq,self.ss):
				st+="\033[44;33m{}\033[m".format(i)
			print(st)
		else:
			print( "Can't highlight. You must pass secondary structure")

	def decode(result,sl):
		#print(result[0])
		wyn=np.argmax(result,axis=-1)
		st=''
		#print(sl)
		#print(wyn)		
		for i in wyn:
			st+=sl[i]
		return st
		
	def __str__(self):
		st=''
		cl={'B':"40",'C':"45",'E':"42",'G':"43",'H':"44",'I':"41",'S': "46",'T':"40" }
		if type(self.ss) == str  and len(self.ss)==len(self.seq):
						
			st='true_seq: \n'
			for i ,j in zip(self.seq,self.ss):
				st+="\033[{};37m{}\033[m".format(cl[j],i)
			return st+'\n'+self.ss
		if type(self.predicted_seq) == str  and len(self.predicted_seq)==len(self.seq):
			if not len(st):
				st+='\npredicted_seq: \n'
			else:
				st+='predicted_seq: \n'
			for i ,j in zip(self.seq,self.predicted_seq):
				st+="\033[{};37m{}\033[m".format(cl[j],i)
			return st +'\n'+self.predicted_seq

		else:
			return self.seq

	def __iter__(self):
		return iter(self.seq)


