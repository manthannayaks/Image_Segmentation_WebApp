import cv2 as cv
import numpy as np
from typing import Tuple

# Global helper function for distance calculation
def pixel_distance(p: np.ndarray, q: np.ndarray) -> float:
    """ Euclidean distance in feature space (Lab or gray). """
    # Calculates: sqrt(sum((p - q)^2))
    return float(np.sqrt(np.sum((p-q)**2)))

# ---------------------------
# Union-Find (Disjoint Set ADT)
# ---------------------------
class UnionFind:
    """
    DSU structure supporting both the Felzenszwalb-Huttenlocher merging criteria
    and the new mean-similarity criteria suggested by the user.
    """
    def __init__(self, n: int, features: np.ndarray = None):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int16)
        self.size = np.ones(n, dtype=np.int32)
        # int_diff is used for the standard F-H merging mode (k parameter)
        self.int_diff = np.zeros(n, dtype=np.float32)
        
        # feature_sum is used for the new mean-based merging mode
        self.feature_sum = None
        if features is not None:
            # Flatten HxWxC features to NxC and store as float32 sums
            self.feature_sum = features.reshape(-1, features.shape[-1]).copy()

    def find(self, x: int) -> int:
        """Finds the representative (root) of the set containing x, with path compression."""
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        
        # Path compression: make all nodes on the path point directly to the root
        while self.parent[x] != x:
            p = self.parent[x]
            self.parent[x] = root
            x = p
        return root

    def _union_sets(self, ra: int, rb: int, w: float):
        """Internal helper to perform the actual union and update DSU properties."""
        # Union by rank
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra  # ensure ra has >= rank
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

        # Update size and int_diff for the new root ra (used by F-H and min_size post-proc)
        self.size[ra] += self.size[rb]
        self.int_diff[ra] = max(self.int_diff[ra], self.int_diff[rb], w)
        
        # Update feature_sum if it exists (used by mean-based merging)
        if self.feature_sum is not None:
            self.feature_sum[ra] += self.feature_sum[rb]

    def union_with_threshold(self, a: int, b: int, w: float, k: float) -> bool:
        """
        Standard F-H merging: Merges based on internal component difference (Tau).
        """
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False

        # Calculate the dynamic thresholds for component Ca and Cb
        thr_a = self.int_diff[ra] + (k / self.size[ra])
        thr_b = self.int_diff[rb] + (k / self.size[rb])
        
        # Check the Felzenszwalb-Huttenlocher merging condition
        if w > min(thr_a, thr_b):
            return False

        self._union_sets(ra, rb, w)
        return True

    def union_by_mean_similarity(self, a: int, b: int, w: float, threshold_mean: float) -> bool:
        """
        New mean-based merging: Merges if the distance between the mean feature
        vectors of the two components is less than a fixed threshold.
        
        The edge weight 'w' (pixel-to-pixel) is still passed to update int_diff 
        for post-processing, but is not used for the merging decision itself.
        """
        if self.feature_sum is None:
            # Should not happen if threshold_mean > 0
            return False 
            
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
            
        # Calculate mean feature vectors for the two components
        mean_a = self.feature_sum[ra] / self.size[ra]
        mean_b = self.feature_sum[rb] / self.size[rb]
        
        # Calculate the distance between the component means
        w_mean = pixel_distance(mean_a, mean_b)
        
        # Check the mean similarity merging condition
        if w_mean > threshold_mean:
            return False

        # Use the strongest edge (w) connecting the components for updating int_diff
        self._union_sets(ra, rb, w)
        return True
        
    def force_union(self, a: int, b: int, w: float):
        """
        Used for post-processing (min_size): unconditionally merge two components.
        """
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        
        # Use the strongest edge (w) connecting the components for updating int_diff
        self._union_sets(ra, rb, w)
        return True
# ---------------------------
# Image helpers (unchanged)
# ---------------------------
def to_grayscale_or_lab(img: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Converts input image to the feature space used for calculating distances.
    For color images, it uses the Lab color space for perceptual uniformity.
    """
    if img.ndim == 2 or img.shape[2] == 1:
        gray = img if img.ndim == 2 else img[:, :, 0]
        return gray[:, :, None].astype(np.float32), False
    else:
        lab = cv.cvtColor(img, cv.COLOR_BGR2Lab).astype(np.float32)
        return lab, True


def build_edges(features: np.ndarray, connectivity: int = 8):
    """
    Builds edges between neighboring pixels. (Unchanged)
    """
    H, W, C = features.shape
    idx = lambda r, c: r * W + c

    edges = []
    
    # Iterate over all pixels and create edges to neighbors (Right, Down, Diagonals)
    for r in range(H):
        for c in range(W):
            p_idx = idx(r, c)
            p_features = features[r, c, :]

            # Right neighbor
            if c + 1 < W:
                w = pixel_distance(p_features, features[r, c + 1, :])
                edges.append((w, p_idx, idx(r, c + 1)))
            
            # Down neighbor
            if r + 1 < H:
                w = pixel_distance(p_features, features[r + 1, c, :])
                edges.append((w, p_idx, idx(r + 1, c)))

            if connectivity == 8:
                # Down-Right diagonal
                if (r + 1 < H) and (c + 1 < W):
                    w = pixel_distance(p_features, features[r + 1, c + 1, :])
                    edges.append((w, p_idx, idx(r + 1, c + 1)))
                
                # Down-Left diagonal
                if (r + 1 < H) and (c - 1 >= 0):
                    w = pixel_distance(p_features, features[r + 1, c - 1, :])
                    edges.append((w, p_idx, idx(r + 1, c - 1)))

    edges.sort(key=lambda t: t[0])
    return edges


def segment_graph(img_bgr: np.ndarray,
                  k: float = 300.0,
                  min_size: int = 50,
                  connectivity: int = 8,
                  resize_to: int = 0,
                  mean_threshold: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segmentation function supporting both F-H (k) and Mean-Similarity (mean_threshold).
    """
    orig_h, orig_w = img_bgr.shape[:2]
    img = img_bgr

    # Optional: resize for speed/scale control
    if resize_to and max(orig_h, orig_w) > resize_to:
        scale = resize_to / float(max(orig_h, orig_w))
        img = cv.resize(img_bgr, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv.INTER_AREA)

    H, W = img.shape[:2]

    # Features (Lab for color, grayscale otherwise)
    features, is_color = to_grayscale_or_lab(img)
    edges = build_edges(features, connectivity=connectivity)

    N = H * W
    
    # Initialize UnionFind based on the chosen mode
    if mean_threshold > 0.0:
        # Mean-based mode: UF needs feature sums
        uf = UnionFind(N, features=features)
        print("--- Using Mean-Similarity Merging Criteria ---")
    else:
        # Standard F-H mode: UF only needs N
        uf = UnionFind(N)
        print("--- Using Felzenszwalb-Huttenlocher Merging Criteria ---")


    # 1. Main merging pass
    for w, u, v in edges:
        if mean_threshold > 0.0:
            # Use the new mean-based merging logic
            uf.union_by_mean_similarity(u, v, w, mean_threshold)
        elif k > 0.0:
            # Use the original F-H merging logic
            uf.union_with_threshold(u, v, w, k)
        # If both k and mean_threshold are 0, no merging happens here (only min_size later)

    # 2. Post-process small components: merge to nearest neighbor (min_size)
    # This step uses the standard F-H post-processing (force_union) regardless of the main mode
    for w, u, v in edges:
        ru, rv = uf.find(u), uf.find(v)
        if ru != rv and (uf.size[ru] < min_size or uf.size[rv] < min_size):
            uf.force_union(u, v, w)

    # 3. Build label map and 4. Visualization (unchanged)
    # ... (rest of the segment_graph is the same)
    # 3. Build label map (compress and renumber roots)
    # Perform final path compression and extract the root for every pixel
    roots = np.fromiter((uf.find(i) for i in range(N)), dtype=np.int32)
    # Map unique roots to new, consecutive labels [0, 1, 2, ...]
    unique_roots, new_labels = np.unique(roots, return_inverse=True)
    labels = new_labels.reshape(H, W).astype(np.int32)

    # 4. Visualization: color each region with its mean color from the input image
    vis = np.zeros_like(img, dtype=np.uint8)
    for lab in range(len(unique_roots)):
        mask = (labels == lab)
        if mask.any():
            # Calculate the mean color (in BGR) of the region
            mean_col = img[mask].mean(axis=0)
            vis[mask] = mean_col.astype(np.uint8)

    # 5. Rescale labels/vis back to original size if resizing was performed
    if (H, W) != (orig_h, orig_w):
        # Use nearest neighbor interpolation for labels to maintain integer boundaries
        labels = cv.resize(labels, (orig_w, orig_h), interpolation=cv.INTER_NEAREST).astype(np.int32)
        # Use nearest neighbor for visualization to maintain sharp boundaries
        vis = cv.resize(vis, (orig_w, orig_h), interpolation=cv.INTER_NEAREST)

    return labels, vis


def random_color_segments(labels: np.ndarray) -> np.ndarray:
    """Quick visualization with random colors per label."""
    H, W = labels.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    n_labels = labels.max() + 1
    # Use a fixed seed for reproducible random colors
    rng = np.random.default_rng(42) 
    palette = rng.integers(0, 255, size=(n_labels, 3), dtype=np.uint8)
    out = palette[labels]
    return out


if __name__ == "__main__":
    # -----------------------
    # CHANGE THIS PATH
    # -----------------------
    path = "coin.png"  # your input image

    img_bgr = cv.imread(path, cv.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image at: {path}. Please check the file path.")

    # --- PARAMETER SETTING ---
    # To use the new Mean-Similarity Merging, set k=0.0 and mean_threshold > 0.0
    # To use the original F-H Merging, set mean_threshold=0.0 and k > 0.0
    
    # Example 1: Use Mean-Similarity Merging (Your new idea)
    k = 0           # Set k to 0 to disable F-H dynamic thresholding
    mean_threshold = 20 # New parameter: Threshold for mean color difference (e.g., 20 in Lab space)
    
    # Example 2 (Original F-H): Uncomment these lines to switch back
    # k = 300.0           
    # mean_threshold = 0.0  

    min_size = 50         # Post-processing: minimum component size (applies to both modes)
    connectivity = 8      # 4 or 8
    resize_to = 800       # speed: cap longest side to 800 px; set 0 to keep original

    # For the single plain color effect you requested:
    # Set mean_threshold to a very high value (e.g., 500) and min_size also high (e.g., 100000)
    # k = 0.0
    # mean_threshold = 500.0
    # min_size = 100000 
    
    print(f"Starting segmentation for {path}...")
    labels, vis = segment_graph(
        img_bgr,
        k=k,
        min_size=min_size,
        connectivity=connectivity,
        resize_to=resize_to,
        mean_threshold=mean_threshold # Pass the new parameter
    )

    # Save outputs
    cv.imwrite("segments_mean_color.png", vis)
    rand_vis = random_color_segments(labels)
    cv.imwrite("segments_random_color.png", rand_vis) 
    
    # Save labels (for ML pipelines)
    np.save("segments_labels.npy", labels)

    print(f"Segmentation complete.")
    print(f"Segments found: {labels.max()+1}")
    print("Saved files: segments_mean_color.png, segments_random_color.png, segments_labels.npy")