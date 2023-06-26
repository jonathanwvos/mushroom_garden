from json import dumps
from math import ceil
from os import listdir
from os.path import join
from random import shuffle
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

import cv2
import numpy as np
import multiprocessing as mp


class MushroomGarden:
    '''
    Wrapper class for all mushroom related logic.
    '''
    
    def __init__(self):
        '''
        Constructor. Create a base path to access mushroom image files, set a
        minimum image size and initiliaze a basic metadata dictionary.
        '''
        self.base_path = join('data', 'species')
        self.min_d = 300 
        self.species = [
            'Agaricus',
            'Amanita',
            'Boletus',
            'Cortinarius',
            'Entoloma',
            'Exidia',
            'Hygrocybe',
            'Inocybe',
            'Lactarius',
            'Russula',
            'Suillus'
        ]

        self.metadata = self.init_metadata()

    def init_metadata(self):
        '''
        Initialize metadata and save the number of images per species.
        '''

        metadata = {}

        for m in self.species:
            m_path = join(self.base_path, m)

            metadata[m] = len(listdir(m_path))

        return metadata
    
    def random_example_set(self):
        '''
        For each species, shuffle image list and return a single image
        from each.
        '''

        for m in self.species:
            mush_path = join(self.base_path, m)
            files = listdir(mush_path)
            shuffle(files)
            
            filename = files[0]
            image_path = join(mush_path, filename)
            bgr_im = cv2.imread(image_path)
            
            yield (m, filename, bgr_im)

    def preprocess_image(self, im: np.ndarray) -> np.ndarray:
        '''
        Resize image so that the longest side matches the minimum length.
        '''

        h,w,_ = im.shape

        if h >= w:
            sf = self.min_d / h
            im = cv2.resize(im, (ceil(w*sf), self.min_d))
        else:
            sf = self.min_d / w
            im = cv2.resize(im, (self.min_d, ceil(h*sf)))

        return im

    def get_sample_set(self, species: str, sample_size: int) -> list:
        '''
        For each species, shuffle image list and return the filenames as a sample set.
        '''

        mush_path = join(self.base_path, species)
        files = listdir(mush_path)
        shuffle(files)
        ss_filenames = files[0:sample_size]

        sample_set = [join(mush_path, f) for f in ss_filenames]

        return sample_set
    
    def avg_euclidean(self, im1: np.ndarray, im2: np.ndarray) -> float:
        '''
        Determine the average euclidean distance between im1 and im2.
        '''

        if im1.shape != im2.shape:
            raise ValueError("The shapes of arr1 and arr2 don't match")
        
        h,w,_ = im1.shape
        dist = 0

        for row in range(h):
            for col in range(w):
                dist += euclidean(im1[row, col], im2[row, col])
        
        return dist/(h*w)

    def process_species_samples(self, species: str, sample_size: int, no_clusters: int, mp_queue) -> float:
        '''
        For each species, retrieve a random sample and calculate the average euclidean
        distance for that species.
        '''

        sample_set = self.get_sample_set(species, sample_size)
        kmeans = KMeans(no_clusters, n_init='auto')

        avg_mush_euclidean = 0
        for ss_filepath in sample_set:
            sample = cv2.imread(ss_filepath)
            sample = self.preprocess_image(sample)
            pixels = self.get_sample_pixels(sample)
            kmeans.fit(pixels)

            cluster_centers = np.rint(kmeans.cluster_centers_).astype('uint8')
            quant = cluster_centers[kmeans.labels_.flatten()]
            quant = quant.reshape(sample.shape)

            avg_mush_euclidean += self.avg_euclidean(sample, quant)

        mp_queue.put(avg_mush_euclidean/sample_size)
    
    def quantize_image(self, filepath: str, no_clusters: int) -> np.ndarray:
        '''
        Retrieve the image associated with filepath and convert to its quantized
        form.
        '''

        kmeans = KMeans(no_clusters, n_init='auto')
        im = cv2.imread(filepath)
        im = self.preprocess_image(im)
        pixels = self.get_sample_pixels(im)
        kmeans.fit(pixels)

        cluster_centers = np.rint(kmeans.cluster_centers_).astype('uint8')
        quant = cluster_centers[kmeans.labels_.flatten()]
        quant = quant.reshape(im.shape)

        return quant

    def process_all_samples(self, sample_size: int, no_clusters: int):
        '''
        Determine the average euclidean distance for all mushrooms by 
        delegating samples to a multiprocessing flow.
        '''

        pool = mp.Pool()
        queue = mp.Manager().Queue()

        for m in self.species:
            pool.apply_async(self.process_species_samples, (m, sample_size, no_clusters, queue))                

        pool.close()
        pool.join()

        result_list = []

        while not queue.empty():
            item = queue.get()
            result_list.append(item)
        
        with open(f'{no_clusters}_results.json', 'w') as f:
            f.write(dumps(
                {
                    'sample_size': sample_size,
                    'avg_mush_euclidean': result_list
                }
            ))

    def get_sample_pixels(self, sample: np.array) -> list:
        '''
        Determine all unique pixels for a sample image.
        '''

        h,w,_ = sample.shape
        pixels = []

        for row in range(h):
            for col in range(w):
                pixels.append(sample[row, col])
        
        return pixels
    