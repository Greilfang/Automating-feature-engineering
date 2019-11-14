## Learning Feature Engineering for Classification

### The Structure

 a stack of ﬁxed-size representations of feature values per target class

passing the $LFE$ classifier requires a a ﬁxed size feature vector representation 

### Problem Formulation

Consider a dataset $D$, with features, $F$=$\{f_1,...,f_n \}$

A target class, ${ \kappa}$

The transformation set  $T=\{T_1,...T_m\}$ and a classification task

A classification task $L$

The problem is to find $q$ best paradigm for constructing new features such that appending to the $D$ to maximizes the accuracy of $L$

Each paradigm consists of a candidate transformation $T_c \in T$ of arity $r$ ,an ordered list of features $[ f_i,...,f_{i+r-1}]$ and a usefulness score.



 

### Transformation Recommendation

LFE models the problem of predicting a useful r-ary transformation $T_c \in T_r$($ T_r\ is\ the\ set\ of\ transformation$) for a given list  $[f_1,...,f_r]$

The input is a  set of features

**one vs rest approach:** 

Each transformation is modelled as a MLP binary classification with real-valued confidence score as output.

Apply the $|T_r|$ MLP on features, if the confidence score > threshold $\theta$ ,then choose this transformation.

$$
c = arg\ max_k(R_{[f_1,...,f_r]})\\
f(n)= \begin{cases} T_c, &\text {if $g_c$($R_{f_1,...f_r}$) }>\theta 
\\ none,& \text{otherwise} \end{cases}
$$

### Feature-Class Representation

The transformation are used to reveal and improve significance correlation or discriminative information between features and class labels. 

Correlation $\uparrow$ the effect $uparrow$

>  *work on improving the correlation!*

LFE represents feature $f$ in a dataset with $k$ classes as follows:

$$
R_f=[Q_f^{(1)};Q_f^{(2)};...;Q_f^{(k)};
$$

$Q_f^{(i)}$ is a fixed-sized representation of values in $f$ that are associate with class$i$  $\to$ $\text {Quatile Sketch Array}$

**The generation of $Q_f^{(i)}$** :**

The high variability in size and range if feature values makes representative learning and PDF learning difficult.

We consider features as high dimensions. We determine a fixed size to capture correlations between features and classes.



**Previous approaches:**

1. Use meta-feature(the distribution of feature to transform the original features.)

2. Fixed-size sampling of both feature and classes. The samples need to reflect the distribution of features.

   Such stratified sampling is difficult to apply on the multiple features with high correlations.
   
3. Feature Hashing can be generated for numerical features but the appropriate function the map small range of feature values to the same hash features is difficult to find.
   
> *Quantile Sketching Array integrate these ways.* 
>
> It is familiar with histogram.
>
> Here we use brute-force way to find k-quantile-point.

**Concrete production of  Quantile Sketch:**

Let $\nu_k$ be the bag of values of feature $f$ for training data point with label $c_k$ and $Q_f^{(i)}$ is the quantile sketch of $\nu_k$

1. Scale values to a predefined range $[lb,wb]$

2. Bucket all values in $\nu_k$ into a set of bins.

3. For example, we partition $[lb,wb]$ into $r$ disjoint bins of uniform width,supposing width $w=\frac{wb-lb}r$

4. Assume the bins are ${b_0,...,b_{r-1}}$,$b_j$ ranges from $[lb+j*w,lb+(j+1)*w]$.

5. function $B(v_l)$ associates the value $v_l$ in $\nu_k$ to the bin $b_j$

   function $P(b_j)$ returns the number of feature values bucketed in $b_j$

6. Finally 
$$
I(b_j)=\frac{P(b_j)}{\sum_{0\leq m<r}{P(b_m)}}
$$

### Training

