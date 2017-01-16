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
from algoritmos import *
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


	def preprocess_calc_distances2(self):
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





	def preprocess_calc_distances3(self):

		self.distances = {}

		vertices = reversed(sorted(self.degreeList.keys()))

		pool = mp.Pool()

		results = []

		for v in vertices:
			logging.info("Chamando método para calcular D ( {} ) para todos os vértices.".format(v))
			r = pool.apply_async(calc_distances_from_v, args=(v,self.degreeList,self.calcUntilLayer,))
			results.append(r)

		for result in results:
			r = result.get()
			logging.info("D ( {} ) para todos os vértices calculadas.".format(r[1]))
			self.distances.update(r[0])


		print self.distances

		preprocess_consolides_distances(self.distances)
		return

	def preprocess_calc_distances(self):

		futures = {}
		distances = {}

		with ProcessPoolExecutor(max_workers = self.workers) as executor:

			for vm,dsm in self.degreeList.iteritems():
				for vd,dsd in self.degreeList.iteritems():
					if(vd > vm):

						distances[vm,vd] = {}

						maxLayer = max(len(dsm),len(dsd))
						for layer in range(0,maxLayer + 1):
							if(layer > self.calcUntilLayer):
								continue
							if (layer in dsm) and (layer in dsd) :
								logging.info("Chamando método para calcular D ( {} , {} ) na camada {}.".format(vm,vd,layer))
								job = executor.submit(fastdtw,dsm[layer],dsd[layer],radius=1,dist=custo)
								futures[job] = (vm,vd,layer)
							else:
								distances[vm,vd][layer] = -1.

						if(len(futures) >= self.simultaneousCalculations):
							logging.info("Recebendo resultados...")

							for job in as_completed(futures):
								dist = job.result()
								r = futures[job]
								logging.info("D ( {} , {} ) na camada {} calculada: {} ".format(r[0],r[1],r[2],dist[0]))
								distances[r[0],r[1]][r[2]] = dist[0]

								del futures[job]


			logging.info("Recebendo resultados...")
			for job in as_completed(futures):
				dist = job.result()
				r = futures[job]
				logging.info("D ( {} , {} ) na camada {} calculada: {} ".format(r[0],r[1],r[2],dist[0]))
				distances[r[0],r[1]][r[2]] = dist[0]

		self.distances = distances

		preprocess_consolides_distances(self.distances)
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


	def simulate_walks(self,num_walks,walk_length):

		with ProcessPoolExecutor(max_workers=self.workers) as executor:
			job = executor.submit(generate_random_walks,num_walks,walk_length,self.workers)

			job.result()

		return		
      	


