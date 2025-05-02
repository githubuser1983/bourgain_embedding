# --- START OF FILE covertree.py ---

import random
import numpy as np # For efficient median finding/partitioning
import warnings
import pickle # For save/load example

# Try importing numba for potential JIT compilation
try:
    import numba
    _numba_available = True
except ImportError:
    _numba_available = False
    # Define dummy decorator if numba not available
    class numba:
        def jit(*args, **kwargs):
            def decorator(func):
                return func
            return decorator

class VPTreeNode:
    """ Node of a vantage-point tree. """
    __slots__ = ('point', 'threshold', 'left', 'right')

    def __init__(self, point, threshold=None, left=None, right=None):
        self.point = point
        self.threshold = threshold
        self.left = left
        self.right = right

# Optional: Apply Numba JIT if operations are numerical
# @numba.jit(nopython=True) # Might work if dist and points are compatible numeric types
def _find_median_distance(vp, points, dist_func):
    """ Helper to find median distance using numpy.partition (O(N) average). """
    if not points:
        return 0.0 # Or handle as error?
    distances = np.array([dist_func(vp, p) for p in points])
    median_idx = len(distances) // 2
    # Partition around the median index - puts median element at median_idx
    np.partition(distances, median_idx)
    return distances[median_idx]

# Class name still technically incorrect, but keeping it as per user file name
class CoverTree:
    """
    A metric nearest-neighbor structure using a Vantage-Point Tree (VP-Tree).
    Provides build(points) and find_nearest_neighbor(query) methods.
    Build uses O(N) median finding.
    """
    def __init__(self, dist):
        if not callable(dist):
            raise TypeError("dist must be a callable metric function")
        self.dist = dist
        self.root = None

    def build(self, points):
        """ Builds the VP-Tree index on the provided list of points. """
        # Make a mutable copy and shuffle for VP selection randomness
        pts = list(points)
        random.shuffle(pts)
        self.root = self._build_recursive(pts)

    # Optional Numba JIT - needs careful testing with actual dist/point types
    # @numba.jit(nopython=True) # Requires dist and point types to be Numba-compatible
    def _build_recursive(self, points):
        """ Recursively build a VP-Tree from points using O(N) median. """
        if not points:
            return None

        # Select vantage point (first after shuffle)
        vp = points[0]

        if len(points) == 1:
            return VPTreeNode(vp) # Leaf node

        # --- O(N) Median Finding ---
        other_points = points[1:]
        if not other_points: # Only VP was left
             return VPTreeNode(vp)

        # Find the actual median distance value efficiently
        # Note: _find_median_distance can be JITted if dist/points are numeric
        threshold = _find_median_distance(vp, other_points, self.dist)
        # threshold = np.median([self.dist(vp, p) for p in other_points]) # Simpler but O(N log N) due to sort in median

        # --- O(N) Partitioning ---
        left_points = []
        right_points = []
        # Include the VP itself in one of the subtrees (here, left)
        # Or handle VP separately? Let's put VP in left based on threshold=0 relative to itself?
        # Simpler: VP is the node, recurse on others partitioned by threshold
        for p in other_points:
            d = self.dist(vp, p)
            if d <= threshold: # Points closer than or equal to median distance
                left_points.append(p)
            else: # Points farther than median distance
                right_points.append(p)

        # Build subtrees recursively
        left_node = self._build_recursive(left_points)
        right_node = self._build_recursive(right_points)

        # Create and return node
        node = VPTreeNode(vp, threshold, left_node, right_node)
        return node

    def find_nearest_neighbor(self, query):
        """ Returns a tuple (nearest_point, distance). """
        if self.root is None:
            return None, float('inf')

        best = [None, float('inf')] # Use list for mutable update in closure

        # Define recursive search function (can be JITted if dist/types allow)
        # @numba.jit(nopython=True) # Decorator here if feasible
        def search(node, current_best_dist_sq): # Pass best dist for potential Numba use
            if node is None:
                return current_best_dist_sq # Return potentially updated best distance

            # Distance to vantage point
            d = self.dist(query, node.point)

            # Update best if current node is closer
            if d < current_best_dist_sq:
                best[0] = node.point # Update external best point via list mutation
                current_best_dist_sq = d

            # Pruning logic
            if node.threshold is None: # Leaf node
                return current_best_dist_sq

            if d < node.threshold:
                # Query point potentially inside the median ball
                # Search nearer (left) side first
                current_best_dist_sq = search(node.left, current_best_dist_sq)
                # Check if farther (right) side needs searching
                if d + current_best_dist_sq >= node.threshold: # Check intersection
                    current_best_dist_sq = search(node.right, current_best_dist_sq)
            else:
                # Query point potentially outside the median ball
                # Search farther (right) side first
                current_best_dist_sq = search(node.right, current_best_dist_sq)
                # Check if nearer (left) side needs searching
                if d - current_best_dist_sq <= node.threshold: # Check intersection
                    current_best_dist_sq = search(node.left, current_best_dist_sq)

            return current_best_dist_sq
        # End of search function

        # Start search, passing initial best distance
        final_best_dist = search(self.root, best[1])
        # best[0] holds the nearest point, final_best_dist holds the distance
        return best[0], final_best_dist


# Example usage (remains the same)
if __name__ == '__main__':
    print("Running VP-Tree (named CoverTree) example...")
    # Simple numeric example
    data = list(range(1000))
    random.shuffle(data)
    X = [(i,) for i in data] # List of tuples

    def euclid_1d(a, b):
        # Numba works well with simple numeric operations
        return abs(a[0] - b[0])

    print("Building tree...")
    start_build = time.time()
    # Use Numba-friendly distance if Numba is installed
    ct = CoverTree(dist=euclid_1d)
    ct.build(X)
    end_build = time.time()
    print(f"Build time: {end_build - start_build:.4f}s")

    print("\nRunning queries...")
    queries = [(10,), (500.2,), (999,), (-5,), (1001,), (45.9,)]
    start_query = time.time()
    for q in queries:
        pt, d = ct.find_nearest_neighbor(q)
        print(f"Query={q}, Nearest={pt}, Distance={d:.4f}")
    end_query = time.time()
    print(f"Total query time: {end_query - start_query:.4f}s")

    # Example with strings (Numba likely won't help much unless dist is specialized)
    print("\nRunning VP-Tree example with strings...")
    try:
        import Levenshtein
        def levenshtein_dist(s1, s2): return Levenshtein.distance(str(s1), str(s2))
        print("Using Levenshtein distance.")
        str_dist = levenshtein_dist
    except ImportError:
        print("Levenshtein not found, using length difference.")
        def len_diff(s1, s2): return abs(len(str(s1)) - len(str(s2)))
        str_dist = len_diff

    str_data = ["apple", "banana", "cherry", "date", "elderberry", "apricot", "blueberry"] * 50
    random.shuffle(str_data)

    ct_str = CoverTree(dist=str_dist)
    print("Building string tree...")
    start_build_str = time.time()
    ct_str.build(str_data)
    end_build_str = time.time()
    print(f"String build time: {end_build_str - start_build_str:.4f}s")

    print("\nRunning string queries...")
    str_queries = ["apples", "banaa", "grape", "apricots"]
    start_query_str = time.time()
    for q in str_queries:
        pt, d = ct_str.find_nearest_neighbor(q)
        print(f"Query='{q}', Nearest='{pt}', Distance={d}")
    end_query_str = time.time()
    print(f"Total string query time: {end_query_str - start_query_str:.4f}s")
