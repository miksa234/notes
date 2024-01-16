# SGD with Large Step Sizes Learns Sparse Features

## WHY?

Setup:

Training data $$ {(x_1, y_1), ..., (x_n, y_n)} \in \R^{d} x Y$$
large-scale ML: large dimension $$d$$, large number of training data $$n$$

Classic examples for fitting the data:

Least squares: $$1/n || Ax - b ||_2^2 =1/n \sum_i^n(a_i^T x - b_i)^2 = 1/n\sum_i^n f_i(x)$$

Support Vector Machine (SVM): $$1/2||x||_2^2 + C/n \sum_i^n max(0, 1 - y_i(x^T a_i + b))$$

Deep Neural Nets $$1/n \sum_i^n loss(y_i, DNN(x; a_i)) = 1/n \sum_i^n f_(x)$$


All of these optimization problems have the common feature $$1/n \sum_i^n f_i(x)$$

So instead of computing at each step the gradient of all the f_i functions we
pick only one (unifrom distribution).

## SGD vs GD

We want to minimize functions of the form $$n \sum_{i}^{n} f_i(x)$$

GD would compute the gradient of every term, updates the next iterate

SGD pics a uniformly random $$ i(r) \in \{1, 2, ..., n\} $$  ->
f_{i(k)} and computes its gradient then updates the next iterate

    x^{k+1} = x^{k} - t_k * \nabla f_{i(r)}(x^k)

Key property : Expectation $$ E[\nabla f_{i(r)}(x)] = \nabla f(x) $$
                is an unbiased estimator !!!

There are two options in choosing:
Randomly pick an index i
    1. with replacement until the epoch is filled each time
    2. without replacement

All toolkits use Option 2.!
All papers use Option 1., because its better for analyzing!

## Large Stepsizes induce Sparse Features?

"sparse" in NN-terminolog meaining that for a feature vector \psi(x)
only a few features are active and others are 0.

Using large step sizes often leads iterates to jump = `loss stabilization`
this is the phase of large step sizes where the loss seams to be on average
constant this induces a hidden dynamics to ward simple predictors

The longer a larger steps size is used the better the implicit regularization
can operate and find sparse representations.

Justification: Theoretically on simple NN models (Diagonal linear networks,
ReLU networks) and qualitatively with
stochastic processes

    Picture!

Residual Network 18 Layers trained on CIFAR-10 (60k 32x32 images) for 100
epochs left training loss, right test error

Two key phases regarding the large step size:
    1. after the start of training loss remains constant (loss stabilization)
    2. despite no progress, running this phase for longer leads to better
       generalization (hidden dynamics)


## To be continued...










#REFS
https://fa.bianp.net/teaching/2018/eecs227at/stochastic_gradient.html
https://www.youtube.com/watch?v=k3AiUhwHQ28
