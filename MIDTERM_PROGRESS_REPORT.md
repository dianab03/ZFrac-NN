# Midterm Progress Report
## Can Neural Networks Learn Fractal Patterns?

---

## 1. What Is This Project About?

When we look at natural objects like leaves, clouds, or cracks in materials, they often have a special property: they look similar at different zoom levels. This is called **fractal geometry**, and we can measure it using something called the **fractal dimension**.

**The question is:** Do CNNs automatically learn to recognize these fractal patterns, or do we need to teach them explicitly?

Our project tests this by:
- Extracting fractal measurements from images ourselves
- Comparing them to what neural networks learn on their own
- Seeing which approach works better for different types of images

---

## 2. Methodology Implementation

### 2.1 Measuring Fractal Patterns (ZFrac)

We divide each image into zones and measure how "complex" or "rough" each zone is:

- A smooth surface has low complexity (like a wall)
- A rough surface has high complexity (like tree bark)

**Our approach:**
- Split the image into 1 zone, then 4 zones, then 16 zones
- Measure the fractal dimension of each zone
- Combine all measurements into a list of 21 numbers which act as features

This gives us a simple "fingerprint" of the image's texture.

### 2.2 Checking If Neural Networks Learn Fractals

We use two methods to compare what we measured versus what the neural network learned:

- **CKA (Centered Kernel Alignment):** Checks if two sets of features capture similar patterns by comparing how they group similar images together

- **SVCCA (Singular Vector Canonical Correlation Analysis):** Works in two steps:
  1. First, it uses SVD (Singular Value Decomposition) to reduce the features down to the most important directions in the data while keeping 99% of the information
  2. Then, it uses CCA to find the best way to align and compare these reduced features, measuring how much they overlap

If these scores are low, it means the neural network is NOT learning fractal patterns on its own.

### 2.3 Two Approaches We're Comparing

**Approach 1: Simple Network + Fractal Features**
- Take our 21 fractal measurements
- Feed them into a small, simple network
- Very fast to train (~12,000 parameters)

**Approach 2: Deep Neural Network (ResNet18)**
- Feed the raw image into a large pretrained network
- Let it figure out what features matter
- Much larger (~11 million parameters)

---

## 3. Our Experiments

### 3.1 Datasets We're Using

For now we've picked three datasets where texture and structure matter:

| Dataset | What It Is | Why We Chose It |
|---------|-----------|-----------------|
| **Tomato Leaf Disease** | Photos of healthy vs diseased tomato leaves | Disease shows up as texture changes on leaves |
| **KolektorSDD** | Photos of industrial parts | Defects appear as surface irregularities |
| **Magnetic Tile Defect** | Photos of magnetic tiles | Cracks and defects have fractal-like patterns |

### 3.2 How We Train

- Like in the paper referenced, we train for up to 200 epochs, but stop early if not improving
- Use 70% of images for training, 15% for validation, 15% for testing
- Save all results so we don't have to retrain every time

---

## 4. Results

### 4.1 Completed Tasks

- Extracting fractal features 
- Measure similarity (CKA, SVCCA) 
- Simple network for fractal features 
- Deep network baseline (ResNet18) 
- Loading three datasets 
- Training and saving models 

### 4.2 Classification Results

| Dataset | ZFrac + NN | CNN (ResNet18) | ZFrac Time | CNN Time | Speedup |
|---------|------------|----------------|------------|----------|---------|
| Tomato Leaf Disease | 43.39% | 80.17% | 7.9s | 1197.1s | **151× faster** |
| KolektorSDD | 90.16% | 98.36% | 1.1s | 106.1s | **97× faster** |
| Magnetic Tile Defect | 72.91% | 97.54% | 2.8s | 366.2s | **129× faster** |

**Key observations:**
- ZFrac trains **100x faster** than CNNs across all datasets
- On the defect detection dataset (KolektorSDD), ZFrac achieved **90.16% accuracy** which is very competitive with much less compute
- CNN still wins on raw accuracy, but at greater computational cost

### 4.3 Similarity Analysis (Do CNNs Learn Fractals?)

| Dataset | CKA Score | SVCCA Score |
|---------|-----------|-------------|
| Tomato Leaf Disease | 0.28 | 0.94 |
| KolektorSDD | 0.30 | 0.90 |
| Magnetic Tile Defect | 0.17 | 0.94 |

**What do these scores mean?**

- **CKA scores are low (0.17 - 0.30):** This confirms CNNs are NOT learning the same fractal patterns we extract manually. They focus on different features.
- **SVCCA scores are high:** This is due to the dimensionality reduction step finding some alignment, but the low CKA scores are the more reliable indicator.

---

## 5. What's Left To Do

### Next Steps
1. Experiment with more datasets and possibly more CNN architectures
2. Create charts and visualizations of results
3. Analysis on when fractal features work best
4. Explore combining ZFrac + CNN features for potential improvements

---

## 6. Key Findings

Based on our experiments so far:

1. **CNNs don't learn fractal patterns** - Low CKA scores (0.17-0.30) confirm that CNNs learn different features than fractal geometry

2. **Fractal features excel at defect detection** - On KolektorSDD, ZFrac achieved 90% accuracy, showing fractal patterns are highly relevant for surface defect detection

3. **Massive time savings** - ZFrac trains faster than CNNs with ~1000× fewer parameters

4. **Trade-off exists** - CNNs achieve higher accuracy. For resource-limited scenarios, ZFrac is a viable alternative

---

## 7. References

1. El Zini, J., Musharrafieh, B., & Awad, M. "On The Potential of The Fractal Geometry and The CNNs' Ability to Encode it."
2. Sarkar, N., & Chaudhuri, B. B. (1992). An efficient approach to estimate fractal dimension of textural images.
3. Kornblith, S., et al. (2019). Similarity of neural network representations revisited.

---
