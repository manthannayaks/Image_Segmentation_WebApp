// segment.js – Browser-optimized graph-based segmentation (Option B)
// Requires OpenCV.js loaded before this script.
// Provides the main function:  segmentImage(matBGR, options)

///////////////////////////////////////////////////////////////////////////////
// Utility Helpers
///////////////////////////////////////////////////////////////////////////////

function clamp(v, min, max) {
    return Math.max(min, Math.min(max, v));
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function mulberry32(a) {
    return function () {
        a |= 0;
        a = a + 0x6D2B79F5 | 0;
        let t = Math.imul(a ^ a >>> 15, 1 | a);
        t = t + Math.imul(t ^ t >>> 7, 61 | a) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }
}

///////////////////////////////////////////////////////////////////////////////
// UNION–FIND (with feature sums + color sums)
///////////////////////////////////////////////////////////////////////////////

class UnionFind {
    constructor(n, C, featureSum) {
        this.n = n;
        this.C = C;

        this.parent = new Int32Array(n);
        this.rank   = new Int16Array(n);
        this.size   = new Int32Array(n);

        this.intDiff = new Float32Array(n);

        for (let i = 0; i < n; i++) {
            this.parent[i] = i;
            this.size[i] = 1;
        }

        // flattened feature sums: LAB features N * C
        this.featureSum = featureSum;
    }

    find(x) {
        let root = x;
        while (this.parent[root] !== root) root = this.parent[root];

        // path compression
        while (this.parent[x] !== x) {
            const p = this.parent[x];
            this.parent[x] = root;
            x = p;
        }
        return root;
    }

    _unionRoots(ra, rb, w) {
        if (ra === rb) return ra;

        if (this.rank[ra] < this.rank[rb]) {
            let tmp = ra; ra = rb; rb = tmp;
        }

        this.parent[rb] = ra;

        if (this.rank[ra] === this.rank[rb]) this.rank[ra]++;

        this.size[ra] += this.size[rb];

        this.intDiff[ra] = Math.max(this.intDiff[ra], this.intDiff[rb], w);

        // merge LAB feature sums
        const baseA = ra * this.C;
        const baseB = rb * this.C;

        for (let c = 0; c < this.C; c++)
            this.featureSum[baseA + c] += this.featureSum[baseB + c];

        return ra;
    }

    unionByMeanSimilarity(a, b, w, thresholdMean) {
        const ra = this.find(a), rb = this.find(b);
        if (ra === rb) return false;

        const baseA = ra * this.C;
        const baseB = rb * this.C;

        const invA = 1.0 / this.size[ra];
        const invB = 1.0 / this.size[rb];

        let sumSq = 0;
        for (let i = 0; i < this.C; i++) {
            const ma = this.featureSum[baseA + i] * invA;
            const mb = this.featureSum[baseB + i] * invB;
            const d  = ma - mb;
            sumSq += d * d;
        }

        const wMean = Math.sqrt(sumSq);

        if (wMean > thresholdMean) return false;

        this._unionRoots(ra, rb, w);
        return true;
    }

    forceUnion(a, b, w) {
        const ra = this.find(a), rb = this.find(b);
        if (ra === rb) return false;
        this._unionRoots(ra, rb, w);
        return true;
    }
}

///////////////////////////////////////////////////////////////////////////////
// Convert cv.Mat -> LAB features
///////////////////////////////////////////////////////////////////////////////

function matToFeatures(mat) {
    const H = mat.rows, W = mat.cols;
    const N = H * W;

    // Convert BGR → LAB
    let lab = new cv.Mat();
    cv.cvtColor(mat, lab, cv.COLOR_BGR2Lab);

    const features = new Float32Array(N * 3);

    let p = 0;
    for (let i = 0; i < lab.data.length; i++) {
        features[p++] = lab.data[i];
    }

    lab.delete();
    return { features, C: 3 };
}

///////////////////////////////////////////////////////////////////////////////
// Build edges (4 or 8 connectivity)
///////////////////////////////////////////////////////////////////////////////

function buildEdges(features, H, W, C, connectivity = 4) {
    const N = H * W;
    const edges = [];

    const idx = (r, c) => r * W + c;
    const offs = (p, ch) => p * C + ch;

    for (let r = 0; r < H; r++) {
        for (let c = 0; c < W; c++) {
            const p = idx(r, c);

            if (c + 1 < W) {
                const q = idx(r, c + 1);
                let sum = 0;
                for (let ch = 0; ch < C; ch++) {
                    const d = features[offs(p, ch)] - features[offs(q, ch)];
                    sum += d * d;
                }
                edges.push({ w: Math.sqrt(sum), u: p, v: q });
            }

            if (r + 1 < H) {
                const q = idx(r + 1, c);
                let sum = 0;
                for (let ch = 0; ch < C; ch++) {
                    const d = features[offs(p, ch)] - features[offs(q, ch)];
                    sum += d * d;
                }
                edges.push({ w: Math.sqrt(sum), u: p, v: q });
            }

            if (connectivity === 8) {
                if (r + 1 < H && c + 1 < W) {
                    const q = idx(r + 1, c + 1);
                    let sum = 0;
                    for (let ch = 0; ch < C; ch++) {
                        const d = features[offs(p, ch)] - features[offs(q, ch)];
                        sum += d * d;
                    }
                    edges.push({ w: Math.sqrt(sum), u: p, v: q });
                }
                if (r + 1 < H && c - 1 >= 0) {
                    const q = idx(r + 1, c - 1);
                    let sum = 0;
                    for (let ch = 0; ch < C; ch++) {
                        const d = features[offs(p, ch)] - features[offs(q, ch)];
                        sum += d * d;
                    }
                    edges.push({ w: Math.sqrt(sum), u: p, v: q });
                }
            }
        }
    }

    edges.sort((a, b) => a.w - b.w);
    return edges;
}

///////////////////////////////////////////////////////////////////////////////
// MAIN SEGMENTATION FUNCTION
///////////////////////////////////////////////////////////////////////////////

async function segmentImage(mat, options = {}) {
    const minSize       = options.minSize ?? 50;
    const connectivity  = options.connectivity ?? 4;
    const meanThreshold = options.meanThreshold ?? 20;

    const H = mat.rows, W = mat.cols;
    const N = H * W;

    const { features, C } = matToFeatures(mat);
    const edges = buildEdges(features, H, W, C, connectivity);

    const featureSum = new Float32Array(features);
    const uf = new UnionFind(N, C, featureSum);

    for (let i = 0; i < edges.length; i++) {
        uf.unionByMeanSimilarity(edges[i].u, edges[i].v, edges[i].w, meanThreshold);
        if ((i & 8191) === 0) await sleep(0);
    }

    for (let i = 0; i < edges.length; i++) {
        const e = edges[i];
        const ru = uf.find(e.u), rv = uf.find(e.v);
        if (ru !== rv && (uf.size[ru] < minSize || uf.size[rv] < minSize)) {
            uf.forceUnion(e.u, e.v, e.w);
        }
        if ((i & 8191) === 0) await sleep(0);
    }

    const roots = new Int32Array(N);
    for (let i = 0; i < N; i++) roots[i] = uf.find(i);

    const map = new Map();
    let label = 0;

    const labels = new Int32Array(N);
    for (let i = 0; i < N; i++) {
        const r = roots[i];
        if (!map.has(r)) map.set(r, label++);
        labels[i] = map.get(r);
    }

    const K = label;

    ///////////////////////////////////////////////////////////////////////////
    // PYTHON-LIKE LAB MEAN COLOR
    ///////////////////////////////////////////////////////////////////////////

    const meanColors = new Uint8ClampedArray(K * 3);

    for (const [root, lab] of map.entries()) {
        const size = uf.size[root] || 1;
        const base = root * C;

        const L = uf.featureSum[base + 0] / size;
        const A = uf.featureSum[base + 1] / size;
        const B = uf.featureSum[base + 2] / size;

        let labMat = new cv.Mat(1, 1, cv.CV_8UC3);
        labMat.data[0] = L;
        labMat.data[1] = A;
        labMat.data[2] = B;

        let bgrMat = new cv.Mat();
        cv.cvtColor(labMat, bgrMat, cv.COLOR_Lab2BGR);

        meanColors[lab * 3 + 0] = bgrMat.data[0];
        meanColors[lab * 3 + 1] = bgrMat.data[1];
        meanColors[lab * 3 + 2] = bgrMat.data[2];

        labMat.delete();
        bgrMat.delete();
    }

    ///////////////////////////////////////////////////////////////////////////
    // RENDER MEAN COLOR CANVAS
    ///////////////////////////////////////////////////////////////////////////

    const meanCanvas = document.createElement("canvas");
    meanCanvas.width = W;
    meanCanvas.height = H;
    const meanCtx = meanCanvas.getContext("2d");
    const meanData = meanCtx.createImageData(W, H);

    let k = 0;
    for (let i = 0; i < N; i++) {
        const lab = labels[i];
        meanData.data[k++] = meanColors[lab * 3 + 2]; // R
        meanData.data[k++] = meanColors[lab * 3 + 1]; // G
        meanData.data[k++] = meanColors[lab * 3 + 0]; // B
        meanData.data[k++] = 255;
    }

    meanCtx.putImageData(meanData, 0, 0);

    ///////////////////////////////////////////////////////////////////////////
    // RANDOM COLOR CANVAS
    ///////////////////////////////////////////////////////////////////////////

    const randomCanvas = document.createElement("canvas");
    randomCanvas.width = W;
    randomCanvas.height = H;
    const randomCtx = randomCanvas.getContext("2d");
    const randomData = randomCtx.createImageData(W, H);

    const rng = mulberry32(42);
    const palette = new Uint8ClampedArray(K * 3);

    for (let i = 0; i < K; i++) {
        palette[i * 3] = rng() * 255;
        palette[i * 3 + 1] = rng() * 255;
        palette[i * 3 + 2] = rng() * 255;
    }

    k = 0;
    for (let i = 0; i < N; i++) {
        const lab = labels[i];
        randomData.data[k++] = palette[lab * 3 + 2];
        randomData.data[k++] = palette[lab * 3 + 1];
        randomData.data[k++] = palette[lab * 3 + 0];
        randomData.data[k++] = 255;
    }

    randomCtx.putImageData(randomData, 0, 0);

    return {
        labels,
        regions: K,
        meanCanvas,
        randomCanvas
    };
}

window.segmentImage = segmentImage;