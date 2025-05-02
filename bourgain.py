# --- START OF FILE covertree.py ---

import math,os
import time
import numpy as np
import pickle, logging
import warnings
import heapq # Often used internally or for verification

# ==============================================================================
# Cover Tree Implementation (Conceptual Structure)
# ==============================================================================
# This section provides a placeholder structure.
# For practical use, install a library: `pip install covertree`
# The actual 'covertree' library handles the complex tree structure internally.

try:
    # Attempt to import the installed library
    from covertree import CoverTree as CoverTreeLibrary
    _covertree_available = True
    print("Using installed 'covertree' library.")
except ImportError:
    warnings.warn("CoverTree library not found. BourgainEmbedding will use brute force. Install with 'pip install covertree'", UserWarning)
    _covertree_available = False
    # Define a dummy class if the library is not installed
    class CoverTreeLibrary:
        def __init__(self, dist=None, root=None):
            self.dist = dist
            self.root = root # Placeholder
            print("WARNING: CoverTree library not found, CoverTree functionality disabled (using dummy).")

        def build(self, points):
            # Dummy build method
            if not points:
                self.root = None
            else:
                # Store points for potential brute-force in dummy methods
                self._points = list(points)
                self.root = 0 # Dummy root representation
            # In a real implementation, this builds the tree structure.

        def find_nearest_neighbor(self, query_point):
            # Dummy NN method - falls back to brute force if needed
            if self.root is None or not hasattr(self, '_points') or not self._points:
                raise ValueError("Cannot find neighbor in an empty or unbuilt dummy tree.")
            if self.dist is None:
                raise ValueError("Distance function not set for dummy tree.")

            min_dist = float('inf')
            nearest_neighbor = None
            for point in self._points:
                d = self.dist(query_point, point)
                if d < min_dist:
                    min_dist = d
                    nearest_neighbor = point
            if nearest_neighbor is None:
                 # Should only happen if _points was empty but root wasn't None
                 raise RuntimeError("Inconsistent dummy tree state.")
            return nearest_neighbor, min_dist

# ==============================================================================
# Bourgain Embedding using Cover Trees
# ==============================================================================

class BourgainEmbedding:
    """
    Implements Bourgain Embedding using Cover Trees (if available) for accelerated
    Exact Nearest Neighbor search to find minimum distances to subsets.
    Falls back to brute force if the 'covertree' library is not installed or
    if Cover Tree operations fail.

    Attributes:
        dist: The distance function `dist(a, b)` between elements of X.
        fast: Whether to use the faster log log n variant for k and T.
        dim: The dimension of the resulting embedding.
        cover_trees_ (dict): Dictionary storing fitted CoverTree instances for each subset.
        subset_indices_ (dict): Dictionary storing the indices of elements in each subset
                                from the original fit data X_fit_.
        X_fit_ (list): Reference to the original data used during fitting.
        n_fit_ (int): Number of data points used in fit.
        k_ (int): Number of levels used in Bourgain embedding.
        T_ (int): Number of trials per level used in Bourgain embedding.
    """
    def __init__(self, dist, fast=False):
        if not callable(dist):
            raise TypeError("The 'dist' argument must be a callable function.")
        self.dist = dist
        self.fast = fast
        self.dim = None
        self._reset_fit_state()

    def _reset_fit_state(self):
        """Resets attributes that store the state of the fitted model."""
        self.cover_trees_ = {}
        self.subset_indices_ = {}
        self.X_fit_ = None
        self.n_fit_ = None
        self.k_ = None
        self.T_ = None

    def _check_is_fitted(self):
        """Checks if the model has been fitted."""
        if self.X_fit_ is None:
             raise RuntimeError("This BourgainEmbedding instance is not fitted yet. Call 'fit' first.")

    def fit(self, X, verbose=False):
        """
        Fits the Bourgain embedding model to the data X. Uses Cover Trees
        for acceleration if available, otherwise uses brute force.

        Args:
            X: The input data (list or array-like).
            verbose: If True, prints progress messages.

        Returns:
            X_emb: The Bourgain embedding of the input data X as a list of lists.
        """
        self._reset_fit_state()
        self.X_fit_ = list(X) # Store data as list
        n = len(self.X_fit_)
        self.n_fit_ = n
        if n == 0:
            self.dim = 0
            warnings.warn("Fitting on empty input X. Embedding dimension is 0.", UserWarning)
            return []

        np.random.seed(123)

        # Determine k and T
        log_n = math.log(n) if n > 1 else 0
        if not self.fast:
            self.k_ = int(math.ceil(log_n / math.log(2) - 1)) if n > 1 else 0
            self.T_ = int(math.ceil(log_n)) if n > 0 else 0
        else:
            log_log_n = math.log(log_n) if n > math.e else 0
            self.k_ = int(math.ceil(log_log_n / math.log(2) - 1)) if n > math.e else 0
            self.T_ = int(math.ceil(log_log_n)) if n > 1 else 0
        self.k_ = max(0, self.k_)
        self.T_ = max(1, self.T_)

        X_emb = [[] for _ in range(n)]
        fit_failed_subsets = 0
        covertree_available_and_used = _covertree_available # Use flag from import

        for i in range(0, self.k_ + 1):
            num_in_subset = 1 << i
            if num_in_subset > n: continue

            for t in range(self.T_):
                key = (i, t)
                S_indices = np.random.choice(n, num_in_subset, replace=False)
                self.subset_indices_[key] = S_indices
                subset_objects = [self.X_fit_[idx] for idx in S_indices]

                # --- Build Cover Tree or mark for brute force ---
                tree = None # Initialize tree as None for this subset
                if covertree_available_and_used and len(subset_objects) > 0:
                    try:
                        # Build CoverTree on the *actual objects* in the subset
                        # Pass the distance function directly
                        tree = CoverTreeLibrary(dist=self.dist)
                        tree.build(subset_objects)
                        self.cover_trees_[key] = tree # Store the built tree
                    except Exception as e:
                        self.cover_trees_[key] = None # Mark as failed on error
                        fit_failed_subsets += 1
                        warnings.warn(f"Failed to build CoverTree for subset ({i},{t}): {e}. Using brute force.", UserWarning)
                        # Keep covertree_available_and_used = True, maybe only this subset failed
                else:
                    self.cover_trees_[key] = None # Cannot build tree if lib not installed or empty subset

                # --- Calculate embedding coordinate for ALL original points ---
                min_distances = np.zeros(n)
                # Get the potentially built tree (could be None)
                current_tree = self.cover_trees_.get(key)

                if current_tree is not None: # Try using Cover Tree
                    try:
                        for j in range(n):
                             query_point = self.X_fit_[j]
                             _, distance = current_tree.find_nearest_neighbor(query_point)
                             min_distances[j] = distance
                    except Exception as e:
                        warnings.warn(f"CoverTree query failed for subset ({i},{t}): {e}. Falling back to brute force.", UserWarning)
                        current_tree = None # Force brute force calculation below
                        fit_failed_subsets += 1
                        # Remove failed tree from storage? Or keep None? Let's keep None.
                        self.cover_trees_[key] = None

                if current_tree is None: # Brute force calculation needed
                    if not subset_objects:
                         min_distances = np.full(n, float('inf'))
                    else:
                         for j in range(n):
                            try:
                                min_distances[j] = min(self.dist(self.X_fit_[j], s_obj) for s_obj in subset_objects)
                            except Exception as dist_err:
                                warnings.warn(f"Distance calculation failed for fit point {j}, subset {key}: {dist_err}. Setting dist to infinity.", RuntimeWarning)
                                min_distances[j] = float('inf')


                # Append coordinate to embeddings
                for j in range(n):
                    X_emb[j].append(min_distances[j])

                if verbose and (t % max(1, self.T_ // 5) == 0 or t == self.T_ - 1):
                     print(f"  Fit: Processed subset {t+1}/{self.T_} for level {i}/{self.k_}")

        if fit_failed_subsets > 0:
            print(f"Warning: CoverTree build/query failed for {fit_failed_subsets} subsets during fit. Brute force was used for those coordinates.")

        self.dim = len(X_emb[0]) if n > 0 else 0
        return X_emb


    def predict(self, X_new, verbose=False):
        """
        Generates Bourgain embeddings for new data points X_new using the fitted model.
        Uses Cover Trees for querying if available and built successfully during fit.

        Args:
            X_new: The new input data (list or array-like).
            verbose: If True, prints progress messages.

        Returns:
            X_emb_new: The Bourgain embedding of X_new as a list of lists.
        """
        #self._check_is_fitted()

        X_new_list = list(X_new)
        n_new = len(X_new_list)
        if n_new == 0:
            return []

        X_emb_new = [[] for _ in range(n_new)]
        query_failed_subsets = 0

        # Iterate through the coordinates consistently using stored keys
        # Using subset_indices_ keys ensures we handle cases where tree build failed
        sorted_keys = sorted(self.subset_indices_.keys())

        #logging.info(f"len(sorted_keys) = {len(sorted_keys)}, {math.log(self.n_fit_)**3}")

        for i, t in sorted_keys:
            key = (i, t)
            tree = self.cover_trees_.get(key) # Get potentially existing tree
            #S_indices = self.subset_indices_.get(key) # Needed for brute force fallback

            min_distances_coord = np.zeros(n_new)

            if tree is not None: # Try using Cover Tree
                try:
                    for j in range(n_new):
                        query_point = X_new_list[j]
                        # The find_nearest_neighbor method of the 'covertree' library
                        # should work directly with the new query point object.
                        _, distance = tree.find_nearest_neighbor(query_point)
                        min_distances_coord[j] = distance
                except Exception as e:
                    import sys
                    
                    logging.warning(f"CoverTree query failed during predict for subset ({i},{t}): {e}. Falling back to brute force for this coordinate.", UserWarning)
                    sys.exit(-1)
                    tree = None # Force brute force
                    query_failed_subsets += 1 # Count fallbacks triggered during predict

            # --- Brute force fallback if tree unavailable or query failed ---
            if tree is None:
                import sys
                sys.exit(-1)
                if S_indices is None or len(S_indices) == 0:
                     min_distances_coord = np.full(n_new, float('inf'))
                else:
                    # Retrieve original objects for the subset from the stored fit data
                    subset_objects = [self.X_fit_[idx] for idx in S_indices]
                    if not subset_objects:
                         min_distances_coord = np.full(n_new, float('inf'))
                    else:
                         for j in range(n_new):
                             try:
                                 min_distances_coord[j] = min(self.dist(X_new_list[j], s_obj) for s_obj in subset_objects)
                             except Exception as dist_err:
                                 warnings.warn(f"Distance calculation failed for predict point {j}, subset {key}: {dist_err}. Setting dist to infinity.", RuntimeWarning)
                                 min_distances_coord[j] = float('inf')

            # Append coordinate results for all new points
            for j in range(n_new):
                X_emb_new[j].append(min_distances_coord[j])

            if verbose and (t % max(1, self.T_ // 5) == 0 or t == self.T_ - 1):
                print(f"  Predict: Processed subset {t+1}/{self.T_} for level {i}/{self.k_}")

        if query_failed_subsets > 0:
            num_coords = len(sorted_keys)
            print(f"Warning: CoverTree query failed for {query_failed_subsets} subset-coordinates during predict. Brute force was used.")

        return X_emb_new

# --- Save/Load Functions ---
# Use pickle to save/load the entire instance, assuming CoverTree objects are pickleable
def save_bourgain_embedding(be, path: str) -> None:
    """ Saves a fitted BourgainEmbedding object using pickle. """
    try:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(be, f)
        print(f"Bourgain embedding (with CoverTrees) saved to: {path}")
    except Exception as e:
        print(f"Error saving Bourgain embedding to {path}: {e}")

def load_bourgain_embedding(path: str):
    """ Loads a BourgainEmbedding object using pickle. """
    try:
        with open(path, "rb") as f:
            be = pickle.load(f)
        if not isinstance(be, BourgainEmbedding):
             warnings.warn(f"Loaded object from {path} might not be a BourgainEmbedding instance.", UserWarning)
        # Check if CoverTree dependency is met if trees were saved
        if hasattr(be, 'cover_trees_') and be.cover_trees_ and not _covertree_available:
            warnings.warn("Loaded Bourgain embedding contains CoverTrees, but the 'covertree' library is not installed. Prediction will use brute-force for those coordinates.", RuntimeWarning)
            # Set trees to None if library isn't available? Or let predict handle it?
            # Let predict handle the fallback for simplicity.

        print(f"Bourgain embedding loaded from: {path}")
        return be
    except FileNotFoundError:
        print(f"Error: Bourgain embedding file not found at {path}")
        raise
    except Exception as e:
        print(f"Error loading Bourgain embedding from {path}: {e}")
        raise


# --- Example Usage ---
if __name__ == "__main__":
    print(f"Running example using Cover Trees (available: {_covertree_available})...")
    # Example non-numeric data
    X_train_obj = ["apple", "banana", "cherry", "date", "elderberry", "apricot", "blueberry", "cantaloupe", "dewberry", "grape", "kiwi", "lemon"] * 20 # 240 strings
    X_test_obj = ["fig", "grapefruit", "honeydew", "avocado", "blackberry", "clementine", "lime", "mango"] # 8 new strings

    try:
        import Levenshtein
        def levenshtein_dist(s1, s2): return Levenshtein.distance(str(s1), str(s2))
        print("Using python-Levenshtein distance.")
        metric_func = levenshtein_dist
    except ImportError:
        print("Warning: python-Levenshtein not found. Using simple length difference.")
        def len_diff_dist(s1, s2): return abs(len(str(s1))-len(str(s2)))
        metric_func = len_diff_dist

    print("\n--- Testing BourgainEmbedding (uses CoverTree if available) ---")
    be_ct = BourgainEmbedding(dist=metric_func, fast=False)

    start_fit = time.time()
    X_emb_train = be_ct.fit(X_train_obj, verbose=False)
    end_fit = time.time()
    fit_method = 'CoverTree' if _covertree_available else 'brute force'
    print(f"Fit time: {end_fit - start_fit:.4f}s (used {fit_method} during fit where possible)")
    print(f"Embedding dimension: {be_ct.dim}")

    start_pred = time.time()
    X_emb_test = be_ct.predict(X_test_obj, verbose=False)
    end_pred = time.time()
    # Predict also uses CoverTree if available and trees were built
    print(f"Predict time: {end_pred - start_pred:.4f}s (used {fit_method} during predict where possible)")

    # Optional: Compare with pure brute force if needed (requires separate class)

    # Save and load test
    save_path = "test_bourgain_embedding_ct.pkl"
    print(f"\nSaving embedding to {save_path}...")
    save_bourgain_embedding(be_ct, save_path)
    print("Loading saved embedding...")
    try:
        be_loaded = load_bourgain_embedding(save_path)

        # Verify loaded embedding predicts the same
        print("Predicting with loaded embedding...")
        X_emb_test_loaded = be_loaded.predict(X_test_obj)

        if be_ct.dim == be_loaded.dim:
             diff = np.linalg.norm(np.array(X_emb_test) - np.array(X_emb_test_loaded))
             print(f"\nNorm of difference between original predict and loaded predict: {diff:.6f}")
             if diff < 1e-9:
                 print("Save/Load successful: Predictions match.")
             else:
                 # Differences might occur if tree build failed for some subsets and brute force was used
                 print(f"Save/Load WARNING: Predictions differ slightly (diff={diff:.6f}). Check warnings during fit/predict.")
        else:
             print("\nDimension mismatch after loading.")

    except Exception as e:
        print(f"Error during load/predict test: {e}")
    finally:
        import os
        if os.path.exists(save_path):
            try: os.remove(save_path); print(f"Removed test file: {save_path}")
            except OSError: pass
