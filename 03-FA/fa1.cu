// rough outline for what we wanna do

// we have Q, K, V
// O matrix for output
// Bc(tile for K/V) and Br(tile for Q)

// for each query, we need
    // m[i] = -infinity   # running max
    // l[i] = 0.0         # running denominator sum


//load tile K and V
    // load tile of Q
        // for each query row in this tile
            // S = Qi[r] * Kj^T  (dot products with all key rows in tile) ---
        
            // do the online softmax thingy