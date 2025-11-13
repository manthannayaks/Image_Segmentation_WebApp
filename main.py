import cv2 as cv
import numpy as np
import math
from typing import Tuple

def distance(a,b):
    return np.sqrt(np.sum((a-b)**2))
#-----------------
# union-find ADT
#-----------------
class UnionFind:
    def __init__(self,n:int,features:np.ndarray=None):

        self.parent=np.arange(n,dtype=np.int32)

        self.rank=np.zeros(n,dtype=np.int16)

        self.size=np.ones(n,dtype=np.int32)

        self.int_diff=np.zeros(n,dtype=np.float32)
        
        self.feature_sum=None

        if features is not None:
        
            self.feature_sum=features.reshape(-1,features.shape[-1]).copy()


    def find(self,x:int)->int:
      
        root=x
        while self.parent[root]!=root:
            root=self.parent[root]
        
       
        while self.parent[x]!=x:
            p=self.parent[x]
            self.parent[x]=root
            x=p
        return root
    

    def _union_sets(self,ra:int,rb:int,w:float):
    
      
        if self.rank[ra]<self.rank[rb]:

            ra,rb=rb,ra  

        self.parent[rb]=ra

        if self.rank[ra]==self.rank[rb]:

            self.rank[ra]+=1

       
        self.size[ra]+=self.size[rb]

        self.int_diff[ra]=max(self.int_diff[ra],self.int_diff[rb],w)
        
     
        if self.feature_sum is not None:

            self.feature_sum[ra]+=self.feature_sum[rb]



    def union_by_mean_similarity(self,a:int, b:int, w:float,threshold_mean:float)->bool:

        if self.feature_sum is None:

            return False 
            
        ra, rb=self.find(a),self.find(b)
        if ra==rb:
            return False
            
   
        mean_a=self.feature_sum[ra]/self.size[ra]
        mean_b=self.feature_sum[rb]/self.size[rb]
        
      
        w_mean=distance(mean_a, mean_b)
        
      
        if w_mean>threshold_mean:
            return False

       
        self._union_sets(ra,rb,w)
        return True
        

    def force_union(self,a:int,b:int, w:float):
        ra,rb =self.find(a), self.find(b)
        if ra==rb:
            return False
        

        self._union_sets(ra,rb,w)
        return True

#---------------------------------------------
# converts to grayscale and LAB space from BGR
#---------------------------------------------       
def is_grayscale(img: np.ndarray) -> bool:
    if img.ndim == 2 or (img.ndim ==3 and img.shape[2] == 1) :
        return True
    return False
def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    if is_grayscale(img):
        gray =img if img.ndim == 2 else img[:,:,0]
    else :
        B=img[:,:,0].astype(np.float32)
        G=img[:,:,1].astype(np.float32)
        R=img[:,:,2].astype(np.float32)
        gray = 0.114*B + 0.587*G + 0.299*R
    gray = gray[:,:,None]
    return gray.astype(np.float32)
def convert_to_lab(img:np.ndarray) ->np.ndarray:
    lab=cv.cvtColor(img,cv.COLOR_BGR2LAB)
    return lab.astype(np.float32)
def to_grayscale_or_lab(img: np.ndarray) -> Tuple[np.ndarray, bool]:
    
    if is_grayscale(img):
        gray_img = convert_to_grayscale(img)
        return gray_img,False
    else:
        lab_img = convert_to_lab(img)
        return lab_img,True

#--------------
#Building edges
#--------------
def build_edges(features:np.ndarray,connectivity:int=8):
    
    H,W,C=features.shape
    idx=lambda r,c:r*W+c

    edges=[]
    
    for r in range(H):
        for c in range(W):
            p_idx=idx(r,c)
            p_features=features[r,c,:]
            
            if c + 1 < W:
                w = np.sqrt(np.sum((p_features - features[r, c + 1, :]) ** 2))
                edges.append((w, p_idx, idx(r, c + 1)))
                
            if r + 1 < H:
                w = np.sqrt(np.sum((p_features - features[r + 1, c, :]) ** 2))
                edges.append((w, p_idx, idx(r + 1, c)))
                
            if connectivity == 8:
                
                if (r + 1 < H) and (c + 1 < W):
                    w = np.sqrt(np.sum((p_features - features[r + 1, c + 1, :]) ** 2))
                    edges.append((w, p_idx, idx(r + 1, c + 1)))
                    
                if (r + 1 < H) and (c - 1 >= 0):
                    w = np.sqrt(np.sum((p_features - features[r + 1, c - 1, :]) ** 2))
                    edges.append((w, p_idx, idx(r + 1, c - 1)))
                    
    edges.sort(key=lambda t: t[0])
    return edges
            
#---------------
#Segmentation
#---------------
def segment_graph(img_bgr: np.ndarray,
                  min_size: int = 50,
                  connectivity: int = 8,
                  resize_to: int = 0,
                  mean_threshold: float = 20) -> Tuple[np.ndarray, np.ndarray]:

    orig_h, orig_w = img_bgr.shape[:2]
    img = img_bgr
    
    if resize_to and max(orig_h, orig_w) > resize_to:
        scale = resize_to / float(max(orig_h, orig_w))
        img = cv.resize(img_bgr, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv.INTER_AREA)

    H, W = img.shape[:2]
    
    features, is_color = to_grayscale_or_lab(img)
    edges = build_edges(features, connectivity=connectivity)

    N = H * W
    
    uf = UnionFind(N, features=features)
    print("--- Using Mean-Similarity Merging Criteria ---")
   
        
    for w, u, v in edges:
        uf.union_by_mean_similarity(u, v, w, mean_threshold)
     

    for w, u, v in edges:
        ru, rv = uf.find(u), uf.find(v)
        if ru != rv and (uf.size[ru] < min_size or uf.size[rv] < min_size):
            uf.force_union(u, v, w)
            
    roots = np.fromiter((uf.find(i) for i in range(N)), dtype=np.int32)
    
    unique_roots, new_labels = np.unique(roots, return_inverse=True)
    labels = new_labels.reshape(H, W).astype(np.int32)

    vis = np.zeros_like(img, dtype=np.uint8)
    for lab in range(len(unique_roots)):
        mask = (labels == lab)
        if mask.any():
            
            mean_col = img[mask].mean(axis=0)
            vis[mask] = mean_col.astype(np.uint8)
            
    if (H, W) != (orig_h, orig_w):
        
        labels = cv.resize(labels, (orig_w, orig_h), interpolation=cv.INTER_NEAREST).astype(np.int32)
        
        vis = cv.resize(vis, (orig_w, orig_h), interpolation=cv.INTER_NEAREST)

    return labels, vis
        
#---------------
#random coloring
#---------------
def random_color_segments(labels: np.ndarray)->np.ndarray:

    H,W = labels.shape

    out = np.zeros((H,W,3),dtype=np.uint8)

    n_labels = labels.max()+1

    rng=np.random.default_rng(42)

    palette = rng.integers(0,255,size=(n_labels,3),dtype=np.uint8)

    out = palette[labels]

    return out

#----------------
#Tesing function
#----------------
def test_color_segmented_graph():

    path = "tiger.png"
    img = cv.imread(path)
    labels,vis = segment_graph(img,min_size=50,connectivity=8,mean_threshold=20)

    random_vis = random_color_segments(labels)
    print("--- Segmentation done ---")

    cv.imshow("original",img)
    cv.imshow("Segmented (mean color)",vis)
    cv.imshow("Segmented (random colors)",random_vis)
    cv.waitKey(0)
    cv.destroyAllWindows()

#test_color_segmented_graph()         #uncomment to test here( make sure to change path in the function )
