## Binary Classification

Hello, and welcome back. In this week we're going to go over the basics of neural network programming. It turns out that when you implement a neural network there are some techniques that are going to be really important. 

For example, if you have a training set of $m$ training examples, you might be use to processing the training set by having a for loop step through your $m$ training examples but it turns out that when you're implementing a neural network, you usually want to process your entire training set without using an explicit for loop to loop over your entire training set. So, you'll see how to do that in this week's materials. 

Another idea: when you organize the computation of a neural network, usually you have what's called a forward pass or forward propagation step, followed by a backward pass or what's called a backward propagation step and so in this week's materials, you also get an intuition about why the computations in learning/in a neural network can be organized in this forward propagation and a separate backward propagation.

For this week's materials, I want to convey these ideas using Logistic Regression in order to make the ideas easier to understand but even if you've seen Logistic Regression before, I think that there'll be some new and interesting ideas for you to pick up in this week's materials. 

So with that, let's get started. 

![alt text](./img/image1.png)

Logistic Regression is an algorithm for binary classification. So let's start by setting up the problem. 

Here's an example of a binary classification problem. You might have an input of an image, like that, and want to output a label to recognize this image as either being a cat, in which case you output 1, or not-cat in which case you output 0, and we're going to use $y$ to denote the output label. 

Let's look at how an image is represented in a computer. To store an image your computer stores three separate matrices corresponding to the red, green, and blue color channels of this image.

So, if your input image is 64 pixels by 64 pixels, then you would have 3 times 64 by 64 matrices corresponding to the red, green and blue pixel intensity values for your images. 

Although to make this little slide I drew these as much smaller matrices, so these are actually 5 by 4 matrices rather than 64 by 64. 

So, to turn these pixel intensity values into a feature vector, what we're going to do is unroll all of these pixel values into an input feature vector $x$. 

So, to unroll all these pixel intensity values into a feature vector, what we're going to do is define a feature vector $x$ corresponding to this image as follows: We're just going to take all the pixel values 255, 231, and so on. 255, 231, and so on until we've listed all the red pixels and then eventually 255,134, 255, 134 and so on until we get a long feature vector listing out all the red, green and blue pixel intensity values of this image. 

If this image is a 64 by 64 image, the total dimension of this vector $x$ will be 64 by 64 by 3 because that's the total numbers we have in all of these matrixes which in this case, turns out to be 12,288, that's what you get if you multiply all those numbers. 

And so we're going to use $n_x=12288$ to represent the dimension of the input features $x$ and sometimes for brevity, I will also just use lowercase $n$ to represent the dimension of this input feature vector. 

So, in binary classification, our goal is to learn a classifier that can input an image represented by this feature vector $x$ and predict whether the corresponding label $y$ is 1 or 0, that is, whether this is a cat image or a non-cat image.

### Notation

![alt text](./img/image2.png)

Let's now lay out some of the notation that we'll use throughout the rest of this course. 

A single training example is represented by a pair, $(x,y)$ where $x$ is an x-dimensional feature vector and $y$, the label, is either 0 or 1. 

Your training set will comprise lower-case $m$ training examples and so your training set will be written $(x^{(1)}, y^{(1)})$ which is the input and output for your first training example $(x^{(2)}, y^{(2)})$ for the second training example up to $(x^{(m)}, y^{(m)})$ which is your last training example and then that altogether is your entire training set. 

So, I'm going to use lowercase $m$ to denote the number of training samples and sometimes to emphasize that this is the number of train examples, I might write this as $m=m_{train}$ and when we talk about a test set, we might sometimes use $m_{test}$ to denote the number of test examples so that's the number of test examples. 

Finally, to output all of the training examples into a more compact notation, we're going to define a matrix capital $X$ as defined by taking you training set inputs $x^{(1)}$, $x^{(2)}$ and so on and stacking them in columns. 

So, we take $x^{(1)}$ and put that as a first column of this matrix, $x^{(2)}$, put that as a second column and so on down to $x^{(m)}$, then this is the matrix capital $X$. 

So, this matrix $X$ will have $m$ columns, where $m$ is the number of training examples and the number of rows or the height of this matrix is $n_x$. 

Notice that in other courses, you might see the matrix capital $X$ defined by stacking up the train examples in rows like so, X1 transpose down to Xm transpose but it turns out that when you're implementing neural networks using this convention I have on the left, will make the implementation much easier. 

So, just to recap, $X$ is a $n_x$ by $m$ dimensional matrix, and when you implement this in Python, you see that X.shape, that's the python command for finding the shape of the matrix, that this an nx, m. That just means it is an nx by $m$ dimensional matrix. 

So, that's how you group the training examples input x into matrix. 

How about the output labels $y$? It turns out that to make your implementation of a neural network easier, it would be convenient to also stack $y$ in columns so we're going to define capital $Y$ to be equal to $y^{(1)}$, $y^{(2)}$, up to $y^{(m)}$ like so. 

So, $Y$ here will be a 1 by $m$ dimensional matrix and again, to use the python notation, the shape of $Y$ will be 1, m which just means this is a 1 by $m$ matrix and as you implement your neural network later in this course you'll find that a useful convention would be to take the data associated with different training examples and by data I mean either x or y, or other quantities you see later but to take the stuff or the data associated with different training examples and to stack them in different columns, like we've done here for both $X$ and $Y$

### Summary

So, that's a notation we'll use for a Logistic Regression and for neural networks networks later in this course. If you ever forget what a piece of notation means, like what is $m$ or what is $n$ or what is something else, we've also posted on the course website a notation guide that you can use to quickly look up what any particular piece of notation means. So with that, let's go on to the next video where we'll start to fetch out Logistic Regression using this notation.

## Logistic Regression

In this video, we'll go over Logistic Regression. This is a learning algorithm that you use when the output labels $y$ in a supervised learning problem are all either zero or one, so for binary classification problems. 

![alt text](./img/image3.png)

Given an input feature vector $x$ maybe corresponding to an image that you want to recognize as either a cat picture or not a cat picture, you want an algorithm that can output a prediction, which we'll call $\hat{y}$, which is your estimate of $y$. 

More formally, you want $\hat{y}$ to be the probability or the chance that, $y=1$ given the input features $x$:

$$ \hat{y} = P(y=1 | x) $$

So, in other words, if $x$ is a picture, as we saw in the last video, you want $\hat{y}$ to tell you, what is the chance that this is a cat picture.

So, $x$, as we said in the previous video, is $n_x$ dimensional vector.

$$ x \in \mathbb{R}^{n_x} $$

Given that the parameters of Logistic Regression will be $w$ which is also an $n_x$ dimensional vector: 

$$ w \in \mathbb{R}^{n_x} $$

together with $b$ which is just a real number:

$$ b \in \mathbb{R} $$

### How can we generate the output $\hat{y}$?

So, given an input $x$ and the parameters $w$ and $b$, how do we generate the output $\hat{y}$? 

#### Incorrect approach to generate $\hat{y}$

Well, one thing you could try, that doesn't work, would be to have 

$$ \hat{y} = w^{T}x + b $$

kind of a linear function of the input $x$. And in fact, this is what you use if you were doing linear regression. 

But this isn't a very good algorithm for binary classification because you want $\hat{y}$ to be the chance that $y=1$

So, $\hat{y}$ should really be between zero and one:

$$
0 \leq \hat{y} \leq 1
$$

and it's difficult to enforce that because 

$$ w^{T}x + b $$ 

can be much bigger than one or it can even be negative, which doesn't make sense for probability that you want it to be between zero and one. 

#### Correct approach to generate $\hat{y}$

So, in Logistic Regression, our output is instead going to be 

$$ \hat{y} = \sigma(w^{T}x + b) $$ 

where $\sigma$ is called the sigmoid function.

This is what the sigmoid function looks like: 

If on the horizontal axis I plot $z$, then the function $\sigma(z)$ looks like this. So, it goes smoothly from zero up to one. 

Let me label my axes here, this is zero and it crosses the vertical axis as 0.5. So, this is what $\sigma(z)$ looks like. 

And we're going to use $z$ to denote this quantity: 

$$ z = w^{T}x + b $$

Here's the formula for the sigmoid function: 

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

where $z$ is a real number

So, notice a couple of things. 

- If $z$ is very large, then $e^{-z}$ will be close to zero. So, then s$\sigma(z)$will be approximately one over one plus something very close to zero, because e to the negative of a very large number will be close to zero. So, this is close to 1. 
And indeed, if you look in the plot on the left, if $z$ is very large, then the $\sigma(z)$ is very close to 1. 

If $z$ is very large, then:

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$ 

$$ \sigma(oo) = \frac{1}{1 + e^{-oo}} = \frac{1}{1 + 0} \approx 1 $$

- Conversely, if $z$ is very small, or it is a very large negative number, then $\sigma(z)$ becomes one over one plus e to the negative Z, and this becomes a huge number. So, this becomes, think of it as one over one plus a number that is very, very big, and so, that's close to zero. 
And indeed, you see that as $z$ becomes a very large negative number, $\sigma(z)$ goes very close to 0. 

If $z$ is very small, then:

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

$$ \sigma(0.000001) = \frac{1}{1 + e^{-0.000001}} = \frac{1}{1 + oo} \approx 0 $$

So, when you implement Logistic Regression, your job is to try to learn parameters $w$ and $b$ so that $\hat{y}$ becomes a good estimate of the chance of $y=1$ 

### Other notation: $\theta$

![alt text](./img/image4.png)

Before moving on, just another note on the notation. 

When we program neural networks, we'll usually keep the parameter $w$ and parameter $b$ separate, where here, $b$ corresponds to an inter-spectrum.

In some other courses, you might have seen a notation that handles this differently. In some conventions, you define an extra feature called $x_0=1$. 

So, that now: $x \in \mathbb{R}^{n_x + 1}$

And then you define:

$$ \hat{y} = \sigma(\theta^{T}x) $$ 

In this alternative notational convention, you have vector parameters $\theta$: $\theta_0$, $\theta_1$, $\theta_2$, down to $\theta_{n_x}$

And so $\theta_0$ plays the role of $b$, that's just a real number, and $\theta_1$ down to $\theta_{n_x}$ play the role of $w$

It turns out, when you implement your neural network, it will be easier to just keep $b$ and $b$ as separate parameters. And so, in this class, we will not use any of this notational convention that I just wrote in red. 

If you've not seen this notation before in other courses, don't worry about it. It's just that for those of you that have seen this notation I wanted to mention explicitly that we're not using that notation in this course. 

But if you've not seen this before, it's not important and you don't need to worry about it. 

### Summary

So, you have now seen what the Logistic Regression model looks like. 

Next, to change the parameters $w$ and $b$ you need to define a cost function. Let's do that in the next video.

Quick question:

What are the parameters of Logistic Regression?

- $w$, an $n_x$ dimensional vector, and $b$, a real number. (Correct)
- $w$ and $b$, both real numbers.
- $w$ and $b$, both $n_x$ dimensional vectors.
- $w$, an identity vector, and $b$, a real number.

## Logistic Regression Cost Function

In the previous video, you saw the Logistic Regression model. 

To train the parameters $w$ and $b$ of a Logistic Regression model, you need to define a cost function. Let's take a look at the cost function you can use to train Logistic Regression.

![alt text](./img/image5.png)

To recap this is what we have defined from the previous slide. 

So, you output $\hat{y}^{(i)} = \sigma(w^Tx^{(i)} + b)$ where $\sigma(z^{(i)}) = \frac{1}{1 + e^{-z^{(i)}}}$ is as defined here. 

So, to learn parameters for your model, you're given a training set of $m$ training examples and it seems natural that you want to find parameters $w$ and $b$ so that at least on the training set, the outputs you have/the predictions you have on the training set, which I will write as $\hat{y}^{(i)}$ that will be close to the ground truth labels $y^{(i)}$ that you got in the training set. 

$$ \hat{y}^{(i)} \approx y^{(i)} $$

So, to fill in a little bit more detail for the equation on top, we had said that $\hat{y}$ is as defined at the top for a training example $x$ and of course for each training example, we're using these superscripts with round brackets with parentheses to index into different train examples.

Your prediction on a training example $i$ which is $\hat{y}^{(i)}$ is going to be obtained by taking the sigmoid function and applying it to $w^Tx^{(i)} + b$

And you can also define $z^{(i)}$ as follows: 

$$ z^{(i)} = w^Tx^{(i)} + b $$

Throughout this course, we will use the notational convention where the superscript parentheses $(i)$ refer to data associated with the i-th training example, whether it is $x^{(i)}$, $y^{(i)}$ or $z^{(i)}$, or any other value linked to that specific example

### The loss function

Now, let's see what loss function or an error function we can use to measure how well our algorithm is doing. 

![alt text](./img/image6.png)

$$ L(\hat{y}, y) = \frac{1}{2} (\hat{y} - y)^2 $$

One thing you could do is define the loss when your algorithm outputs $\hat{y}$ and the true label is $y$ to be maybe the square error or one half a square error. It turns out that you could do this, but in Logistic Regression people don't usually do this because when you come to learn the parameters, you find that the optimization problem, which we'll talk about later becomes non convex so you end up with optimization problem, you're with multiple local optima. 

So, Gradient Descent may not find a global optimum. If you didn't understand the last couple of comments, don't worry about it, we'll get to it in a later video. 

But the intuition to take away is that this function $L$ called the loss function is a function will need to define to measure how good our output $\hat{y}$ is when the true label is $y$ and squared error seems like it might be a reasonable choice except that it makes Gradient Descent not work well. 

So, in Logistic Regression were actually define a different loss function that plays a similar role as squared error but will give us an optimization problem that is convex and so we'll see in a later video becomes much easier to optimize

So, what we use in Logistic Regression is actually the following loss function:

$$ L(\hat{y}, y) = - [ y * log(\hat{y}) + (1-y) * log(1-\hat{y}) ] $$

Here's some intuition on why this loss function makes sense: Keep in mind that if were using squared error then you want to square error to be as small as possible and with this Logistic Regression loss function will also want this to be as small as possible. 

To understand why this makes sense, let's look at the two cases.

#### 1st case: $y = 1$

In the first case let's say $y=1$, then the loss function becomes

$$ L(\hat{y}, y) = -log(\hat{y}) $$

So, this says if $y=1$, you want negative log $\hat{y}$ to be as small as possible so that means you want log $\hat{y}$ to be large to be as big as possible, and that means you want $\hat{y}$ to be large but because $\hat{y}$ is you know the sigmoid function, it can never be bigger than one so this is saying that if $y=1$, you want, $\hat{y}$ to be as big as possible, but it can't ever be bigger than one. 

So saying you want, $\hat{y}$ to be close to one as well

#### 2nd case: $y = 0$

If $y = 0$, 

$$ L(\hat{y}, y) = -log(1 - \hat{y}) $$

and so if in your learning procedure you try to make the loss function small what this means is that you want, Log 1 minus $\hat{y}$ to be large and because it's a negative sign there. 

And then through a similar piece of reasoning, you can conclude that this loss function is trying to make $\hat{y}$ as small as possible, and again, because $\hat{y}$ has to be between zero and 1, this is saying that if $y=0$, then your loss function will push the parameters to make $\hat{y}$ as close to zero as possible. 

Now, there are a lot of functions with roughly this effect that if y is equal to one, try to make $\hat{y}$ large and y is equal to zero or try to make $\hat{y}$ small. 

We just gave here in green a somewhat informal justification for this particular loss function. We will provide an optional video later to give a more formal justification for why in Logistic Regression, we like to use the loss function with this particular form. 

Finally, the last function was defined with respect to a single training example. It measures how well you're doing on a single training example

### The cost function

I'm now going to define something called the cost function, which measures how well you doing on the entire training set.

![alt text](./img/image7.png)

So, the cost function $J$, which is applied to your parameters $w$ and $b$, is going to be the average, really one over $m$ of the sum of the loss function applied to each of the training examples in turn.

$$ J(w, b) = \frac{1}{m} \sum_{i=1}^{m} L(\hat{y}^{(i)}, y^{(i)}) $$

$\hat{y}$ is of course the prediction output by your Logistic Regression algorithm using a particular set of parameters $w$ and $b$. 

And so just to expand this out, this is: 

$$ J(w, b) = - \frac{1}{m} \sum_{i=1}^{m} [ y^{(i)} * log(\hat{y}^{(i)}) + (1 - y^{(i)}) * log(1 - \hat{y}^{(i)}) ] $$

So, the minus sign is outside everything else. 

So, the terminology I'm going to use is that the loss function is applied to just a single training example and the cost function is the cost of your parameters, so in training your Logistic Regression model, we're going to try to find parameters $w$ and $b$ that minimize the overall cost function $J$

### Summary

So, you've just seen the setup for the Logistic Regression algorithm, the loss function for training example and the overall cost function for the parameters of your algorithm. 

It turns out that Logistic Regression can be viewed as a very, very small neural network. 

In the next video, we'll go over that so you can start gaining intuition about what neural networks do. 

So, with that let's go on to the next video about how to view Logistic Regression as a very small neural network.

Quick question:

What is the difference between the cost function and the loss function for logistic regression?

- The loss function computes the error for a single training example; the cost function is the average of the loss functions of the entire training set. (Correct)

- They are different names for the same function.

- The cost function computes the error for a single training example; the loss function is the average of the cost functions of the entire training set.

## Gradient Descent



## Derivatives

## More Derivative Examples

## Computation Graph

## Derivatives with a Computation Graph

## Logistic Regression Gradient Descent

## Gradient Descent on $m$ Examples

## Derivation of DL/dz (Optional)