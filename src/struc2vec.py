# -*- coding: utf-8 -*-

import numpy as np
import random,sys,logging,gc
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from multiprocessing import Manager
from time import time
from fastdtw import fastdtw
#import pympler.asizeof as asizeof
import cPickle as pickle
from collections import deque
from collections import defaultdict
import gc

### import local
from utils import *
from algorithms import *
from algorithms_distances import *
import graph



class Graph():
	def __init__(self, g, is_directed, workers, simultaneousCalculations = 20000, calcUntilLayer = 2, fractionCalcDists = 0.2):
		self.G = g
		self.is_directed = is_directed
		self.workers = workers
		self.calcUntilLayer = calcUntilLayer
		self.fractionCalcDists = fractionCalcDists
		self.simultaneousCalculations = simultaneousCalculations


	def get_diameter(self):
		self.diameter = restoreVariableFromDisk('diameter')

		logging.info('Diâmetro do grafo: {}'.format(self.diameter))

		
	def calc_diameter(self):

		with ProcessPoolExecutor(max_workers=self.workers) as executor:
			job = executor.submit(getDiameter,self.G)
			
			job.result()

		return

	def preprocess_neighbors_with_bfs(self):

		with ProcessPoolExecutor(max_workers=self.workers) as executor:
			job = executor.submit(exec_bfs,self.G,self.workers)
			
			job.result()

		return


	def preprocess_calc_distances(self):
		futures = {}
		distances = {}

		vertices = reversed(sorted(self.G.keys()))

		logging.info("Recuperando degreeList do disco...")
		degreeList = restoreVariableFromDisk('degreeList')


		# Usando um dicionário só para todos os processos, é mais lento que criar uma cópia para cada processo
		manager = Manager()
		degreeListP = manager.dict()
		degreeListP.update(degreeList)

		with ProcessPoolExecutor(max_workers = self.workers) as executor:

			for v in vertices:
				logging.info("Chamando método para calcular D ( {} ) para todos os vértices.".format(v))
				job = executor.submit(calc_distances_from_v,v,degreeListP,self.calcUntilLayer)
				futures[job] = v


			logging.info("Recebendo resultados...")

			for job in as_completed(futures):
				dists = job.result()
				r = futures[job]
				logging.info("D ( {} ) para todos os vértices calculadas.".format(r))
				distances.update(dists)

		preprocess_consolides_distances(distances)
		logging.info("Salvando distâncias no disco...")
		saveVariableOnDisk(distances,'distances')
		return


	def preprocess_calc_distances_with_vertices(self, vertices, layer):

		futures = {}
		distances = {}

		parts = self.workers

		logging.info("Recuperando degreeList do disco...")
		degreeList = restoreVariableFromDisk('degreeList')
		# Usando um dicionário só para todos os processos, é mais lento que criar uma cópia para cada processo
		manager = Manager()
		degreeListP = manager.dict()
		degreeListP.update(degreeList)


		vertices = list(vertices)
		chunks = partition(vertices,parts)

		with ProcessPoolExecutor(max_workers = self.workers) as executor:

			part = 1
			for c in chunks:
				job = executor.submit(calc_distances_with_list,c,degreeListP,layer)
				futures[job] = part
				part += 1


			logging.info("Recebendo resultados...")
			for job in as_completed(futures):
				dist = job.result()
				r = futures[job]
				logging.info("Parte {} do cálculo das distâncias da camada {} calculada. ".format(r,layer))
				distances.update(dist)
				#updateDistances(self.distances,dist,layer)

		
		futures = {}
		with ProcessPoolExecutor(max_workers = 1) as executor:
			job = executor.submit(consolidesDistances,distances,layer)
			futures[job] = layer

		for job in as_completed(futures):
			dist = job.result()
			logging.info("Distâncias da camada {} salvas no disco. ".format(layer))


		return


	def preprocess_calc_distances_with_threshold(self):
		
		for layer in range(self.calcUntilLayer + 1, self.diameter + 1):
			logging.info('Calculando distâncias para a camada {}...'.format(layer))

			futures = {}
			vertices_selected = []

			with ProcessPoolExecutor(max_workers = self.workers) as executor:
				job = executor.submit(selectVertices,layer,self.fractionCalcDists)
				vertices_selected = job.result()


			self.preprocess_calc_distances_with_vertices(vertices_selected,layer)

			logging.info('Distâncias calculadas para a camada {}.'.format(layer))



	def create_distances_network(self):

		with ProcessPoolExecutor(max_workers=self.workers) as executor:
			job = executor.submit(generate_distances_network,self.diameter)

			job.result()

		return

	def preprocess_parameters_random_walk(self):

		with ProcessPoolExecutor(max_workers=self.workers) as executor:
			job = executor.submit(generate_parameters_random_walk)

			job.result()

		return


	def simulate_walks(self,num_walks,walk_length):

		#generate_random_walks(num_walks,walk_length,self.workers,self.diameter)

		with ProcessPoolExecutor(max_workers=self.workers) as executor:
			job = executor.submit(generate_random_walks,num_walks,walk_length,self.workers,self.diameter)

			job.result()

		return	

	def simulate_walk(self,visits_node):

		generate_random_walk(visits_node,self.diameter)

		# with ProcessPoolExecutor(max_workers=self.workers) as executor:
		# 	job = executor.submit(generate_random_walks,visits_node,self.diameter)

		# 	job.result()

		# return	

	def get_ramdom_walks(self):
		logging.info("Recuperando RWs do disco...")
		rws = restoreVariableFromDisk('random_walks')
		logging.info("RWs recuperadas.")
		return rws

	def get_ramdom_walks_balls(self):
		logging.info("Recuperando RWs do disco...")
		rws = restoreVariableFromDisk('random_walks_balls')
		logging.info("RWs recuperadas.")
		return rws

	def create_walks_from_balls(self,walk_length_balls):

		#generate_random_walks_balls(self.G,self.workers,walk_length_balls)

		with ProcessPoolExecutor(max_workers=self.workers) as executor:
			job = executor.submit(generate_random_walks_balls,self.G,self.workers,walk_length_balls)

			job.result()

		return

	def calcSpectralGap(self):
		calcSpectralGap()

		

      	


