""" 
@file: AutocalibrationSolver.py
@description:   python class to autocalibrate anchor position based on inter-anchor ranging data
                This process takes an initial anchors coords guess as starting point of the iterative
                optimization.
@author: Esau Ortiz
@date: October 2021
"""

import numpy as np
from scipy.optimize import fmin

class AutocalibrationSolver(object):
    def __init__(self, autocalibration_samples, initial_guess, fixed_anchors, max_iters = 1500, convergence_thresh = 0.01, LSq_min_anchors = 4, lower_percentile = 0.25, upper_percentile = 0.75, verbose = False):
        """ AutocalibrationSolver is a multi-stage procedure to autocalibrate
            anchor coordinates based on inter-anchor ranges
        Parameters
        ----------
        autocalibration_samples: (N, M, N) array
            inter-anchor ranges (e.g. autocalibration_samples(0,:,1) contains M ranges between anchor_0 and anchor_1)
        initial_guess: (N, 3) array
            initial guess of anchor coordinates
        autocalibrated_coords: (N, 3) array
            current autocalibrated anchor coordinates
        fixed_anchors: (N, ) bool array
            bool mask of anchors whose position is assumed to be fixed
        max_iters: int
            Stage 1 maximum number of iterations
        convergence_thresh: float
            maximum difference between inter-anchor distances between stage 1 iterations 't' and 't-1'
            to consider that stage 1 has converged
        LSq_min_anchors: int
            minimum number of anchors to perform a LSq anchor coordinates optimization (coordinatesOpt method)
        lower_percentile: float
            inter-anchor range percentile to filter M samples. If a given sample 'm' `range_m` between anchors `anchor_0` and `anchor_1` satisfies range_m >= np.percentile(autocalibration_samples[0,1,:], lower_percentile) is considered in Stage 2 (costOpt method)
        upper_percentile: float
            same as lower percentile but this time is upper limit
        """
        self.samples_ijk = autocalibration_samples
        self.initial_guess = initial_guess
        self.autocalibrated_coords = initial_guess
        self.fixed_anchors = fixed_anchors
        self.max_iters = max_iters
        self.convergence_thresh = convergence_thresh
        self.LSq_min_anchors = LSq_min_anchors
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.verbose = verbose

    def preconditioner(self, samples_ik):
        """ Turn samples_ik into a symmetric matrix
            as ideally it should be
        Parameters
        ----------
        samples_ik: (N, N) array
            inter-anchors ranges array
        Returns
        -------
        samples_ik: (N, N) array
            symmetric inter-anchors ranges array
        """
        n_anchors, _ = samples_ik.shape
        max_diff = -1
        for i in range(n_anchors):
            for k in range(n_anchors):
                _range = samples_ik[i,k]
                _sym_range = samples_ik[k,i]
                # complete unread range if one of the symmetric ranges has been measured
                if _range < 0.0 and _sym_range > 0.0: samples_ik[i,k] = samples_ik[k,i]
                elif _range > 0.0 and _sym_range < 0.0: samples_ik[k,i] = samples_ik[i,k]
                # average if both symmetric ranges are different (actually it if performed always leaving same range if both are equal)
                samples_ik[i,k] = samples_ik[k, i] = np.mean((samples_ik[i,k], samples_ik[k,i]))
                diff = _range - _sym_range
                if _range > 0 and _sym_range > 0 and np.abs(diff) > max_diff: max_diff = diff

        return samples_ik

    def stageOne(self, sample_idx = None):
        """ Stage 1 of multi-stage procedure
        Parameters
        ----------
        sample_idx (option): int
            sample index for which the stage 1 is performed
            if it is not provided the median will be computed
        """
        n_anchors, _, _ = self.samples_ijk.shape
        # if sample_idx is provided stageOne is performed for that index and for the median otherwise
        if sample_idx is None: 
            # compute median (n_anchors, n_anchors) array discarding bad ranges (i.e. range = -1.0)
            samples_ik = np.empty((n_anchors, n_anchors), dtype = float)
            for i in range(n_anchors):
                for k in range(n_anchors):
                    samples = self.samples_ijk[i,:,k][self.samples_ijk[i,:,k] > 0]
                    if samples.shape[0] > 0: samples_ik[i,k] = np.median(samples)
                    else: samples_ik[i,k] = -1.0
        else: 
            samples_ik = np.copy(self.samples_ijk[:,sample_idx,:])

        #samples_ik = self.preconditioner(samples_ik)

        for _ in range(self.max_iters):
            # save previous anchors coords for termination condition
            autocalibrated_coords_old = np.copy(self.autocalibrated_coords)

            # update anchors coords
            for i in range(n_anchors):
                # don't update fixed anchor
                if self.fixed_anchors[i] == True: continue
                _anchors_coords = []
                _ranges = []
                for k in range(n_anchors):
                    # skip if range has not been received (i.e. range  == -1)
                    if samples_ik[i,k] < 0.0: continue
                    _anchors_coords.append(self.autocalibrated_coords[k])
                    _ranges.append(samples_ik[i,k])

                if len(_anchors_coords) >= self.LSq_min_anchors:
                    _anchors_coords = np.array(_anchors_coords)
                    _ranges = np.array(_ranges)
                    self.autocalibrated_coords[i] = AutocalibrationSolver.coordinatesOpt(_anchors_coords, _ranges)

            # termination criterion -> autocalibrated coords have not been modified significantly
            if np.abs(np.linalg.norm(self.autocalibrated_coords - autocalibrated_coords_old)) < self.convergence_thresh:
                break


    def stageTwo(self, sample_idx = None):
        """ Stage 2 of multi-stage procedure
        Parameters
        ----------
        sample_idx (option): int
            sample index for which the stage 2 is performed
            if it is not provided the median will be computed
        """
        n_anchors, _, _ = self.samples_ijk.shape
        # if sample_idx is provided stageOne is performed for that index and for the median otherwise
        if sample_idx is None: 
            # compute median (n_anchors, n_anchors) array not taking into account bad ranges (i.e. range = -1.0)
            _samples_ik = -np.ones((n_anchors, n_anchors), dtype = float)
            for i in range(n_anchors):
                for k in range(n_anchors):
                    samples = self.samples_ijk[i,:,k][self.samples_ijk[i,:,k] > 0]
                    if samples.shape[0] > 0: _samples_ik[i,k] = np.median(samples)
        else:  
            samples_ik = np.copy(self.samples_ijk[:,sample_idx,:])
            # build upper_bounds and lower_bounds matrices
            upper_bounds = -np.ones((samples_ik.shape), dtype = float)
            lower_bounds = -np.ones((samples_ik.shape), dtype = float)
            for i in range(n_anchors):
                for k in range(n_anchors):
                    samples = self.samples_ijk[i,:,k][self.samples_ijk[i,:,k] > 0.0]
                    if samples.shape[0] > 0:
                        lower_bounds[i, k] = np.percentile(samples, self.lower_percentile)
                        upper_bounds[i, k] = np.percentile(samples, self.upper_percentile)
            mask1 = samples_ik <= upper_bounds
            mask2 = samples_ik >= lower_bounds
            mask = mask1 & mask2
            # filter ranges outside limits
            _samples_ik = -np.ones(samples_ik.shape)
            _samples_ik[mask] = samples_ik[mask]

        #_samples_ik = self.preconditioner(_samples_ik)
        # optimization based on scipy.optimize.fmin
        self.autocalibrated_coords =  AutocalibrationSolver.costOpt(self.autocalibrated_coords, _samples_ik, self.fixed_anchors, self.verbose)

    def estimationError(self, gt, est = None, axis = None):
        """ Return estimation error computed as euclidean
        distance
        Parameters
        ----------
        gt: (N, 3) array
            ground truth anchor coordinates
        est (optional): (N, 3) array
            estimated anchor coordinates
        axis (optional): int
            if axis is provided error in given axis
            is returned
        Returns
        -------
        error: (N, ) array
            euclidean distance between ground truth and 
            estimation
        """
        if est is not None:
            self.autocalibrated_coords = est
        if axis is not None:
            est_error = self.autocalibrated_coords - gt
            return est_error[:, axis]
        else:
            return np.sqrt(np.einsum("ijk->ij", (self.autocalibrated_coords[:, None, :] - gt) ** 2))

    @staticmethod
    def coordinatesOpt(anchors_coords, ranges):
            """ Least squares optimization 
            Parameters
            ----------
            anchors_coords: (N, 3) array
                anchor coordinates
            ranges: (N, ) array
                anchor-tag range
            """
            # build A matrix
            A = 2 * np.copy(anchors_coords)
            for i in range(A.shape[0] - 1): A[i] = A[-1] - A[i]
            A = A[:-1] # remove last row

            # build B matrix
            B = np.copy(ranges)**2
            B = B[:-1] - B[-1] - np.sum(anchors_coords**2, axis = 1)[:-1] + np.sum(anchors_coords[-1]**2, axis = 0)
            
            # solve LS and return
            return np.dot(np.linalg.pinv(A), B)    

    @staticmethod
    def costOpt(anchors_coords, ranges_ik, fixed_anchors, verbose = False):
        """ Cost optimization based on scipy.optimize.fmin 
        Parameters
        ----------
        anchors_coords: (N, 3) array
            anchor coordinates
        ranges_ik: array (N, N) 
            inter anchor ranges
        fixed_anchors: (N, ) array
            bool mask of fixed anchors
        Returns
        -------
        Theta_opt: (N, 3) array
            optimized anchor coordinates
        """
        def _my_opt_func(Theta, *args):
            """ Optimize target function
            Parameters
            ----------
            Theta: (N, 3)
                current array of anchors coordinates (x,y,z)
            args:
                ranges_ik: array (N, N) 
                    inter anchor ranges
                n_anchors: int
                    total number of anchors
                fixed_anchors: (N, ) array
                    bool mask of fixed anchors
                Theta_init: (N, 3) array
                    initial anchor coordinates
            Returns
            -------
            cost: float
            """
            ranges_ik, n_anchors, fixed_anchors, Theta_init = args
            Theta = Theta.reshape(n_anchors, 3)

            # Modify current Theta to keep anchors fixed
            Theta[fixed_anchors] = Theta_init[fixed_anchors]
            # compute cost
            distances_ij = np.einsum("ijk->ij", (Theta[:, None, :] - Theta) ** 2)
            cost_ij = (distances_ij - ranges_ik ** 2) ** 2
            # remove j = i costs
            cost_ij[distances_ij == 0] = 0
            # remove costs computed with invalid ranges (i.e. ranges < 0)
            cost_ij[ranges_ik < 0.0] = 0
            return np.sum(np.einsum("ij->i", cost_ij))

        Theta_init = np.copy(anchors_coords)
        n_anchors = anchors_coords.shape[0]
        args = ranges_ik, n_anchors, fixed_anchors, Theta_init
        
        if verbose: print(f'Before optimization: Cost = {_my_opt_func(anchors_coords, *args)}')
        Theta_opt = fmin(_my_opt_func, anchors_coords, args = args, disp=False)    
        if verbose: print(f'After optimization: Cost = {_my_opt_func(Theta_opt, *args)}')
        Theta_opt = Theta_opt.reshape(anchors_coords.shape)
        Theta_opt[fixed_anchors] = Theta_init[fixed_anchors]

        return Theta_opt