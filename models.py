from cmath import inf
import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        activation = nn.DotProduct(self.get_weights(), x)
        return activation

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        activation = nn.as_scalar(self.run(x))
        if activation < 0:
            return -1
        return 1


    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        changed = True
        while changed == True:
            changed = False
            for (x,y) in dataset.iterate_once(1):
                classification = self.get_prediction(x)
                if nn.as_scalar(y) != classification:
                    self.w.update(x,nn.as_scalar(y))
                    changed = True

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.numNeurons = 100
        # self.apha = 0.5
        # self.layers = 1
        self.w1 = nn.Parameter(1,self.numNeurons)
        self.b1 = nn.Parameter(1,self.numNeurons)
        self.w2 = nn.Parameter(self.numNeurons,1)
        self.b2 = nn.Parameter(1,1)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        xw = nn.Linear(x,self.w1)
        xwb = nn.AddBias(xw,self.b1)
        z = nn.ReLU(xwb)
        sum = nn.Linear(z,self.w2)
        predicted_y = nn.AddBias(sum,self.b2)
        return predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        loss = nn.SquareLoss(predicted_y,y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batchsize = 200
        learningRate = -1 * 0.01
        scalarLoss = inf
        while scalarLoss > 0.02:
            for x, y in dataset.iterate_once(batchsize):
                loss = self.get_loss(x,y)
                gw,gb,gw2,gb2 = nn.gradients(loss,[self.w1,self.b1,self.w2,self.b2])
                self.w1.update(gw,learningRate)
                self.b1.update(gb,learningRate)
                self.w2.update(gw2,learningRate)
                self.b2.update(gb2,learningRate)
            newLoss = self.get_loss(x,y)
            scalarLoss = nn.as_scalar(newLoss)
        return


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.numNeurons = 200

        self.w1 = nn.Parameter(784,self.numNeurons)
        self.b1 = nn.Parameter(1,self.numNeurons)

        self.w2 = nn.Parameter(self.numNeurons,self.numNeurons)
        self.b2 = nn.Parameter(1,self.numNeurons)

        self.w3 = nn.Parameter(self.numNeurons,self.numNeurons)
        self.b3 = nn.Parameter(1,self.numNeurons)
        
        self.wf = nn.Parameter(self.numNeurons,10)
        self.bf = nn.Parameter(1,10)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        xw1 = nn.Linear(x,self.w1)
        xwb1 = nn.AddBias(xw1,self.b1)
        z1 = nn.ReLU(xwb1)

        xw2 = nn.Linear(z1,self.w2)
        xwb2 = nn.AddBias(xw2,self.b2)
        z2 = nn.ReLU(xwb2)

        xw3 = nn.Linear(z2,self.w3)
        xwb3 = nn.AddBias(xw3,self.b3)
        z3 = nn.ReLU(xwb3)

        sum = nn.Linear(z3,self.wf)
        predicted_y = nn.AddBias(sum,self.bf)
        return predicted_y


        # sum = nn.Linear(z2,self.wf)
        # predicted_y = nn.AddBias(sum,self.bf)
        # return predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        loss = nn.SoftmaxLoss(predicted_y,y)
        return loss


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batchsize = 60
        learningRate = -1 * 0.5
        validationAccuracy = 0
        epoch = 0
        while validationAccuracy < 0.98 and epoch < 11:
            for x, y in dataset.iterate_once(batchsize):
                loss = self.get_loss(x,y)
                gw,gb,gw2,gb2,gwf,gbf = nn.gradients(loss,[self.w1,self.b1,self.w2,self.b2,self.wf,self.bf])
                self.w1.update(gw,learningRate)
                self.b1.update(gb,learningRate)

                self.w2.update(gw2,learningRate)
                self.b2.update(gb2,learningRate)

                self.wf.update(gwf,learningRate)
                self.bf.update(gbf,learningRate)
            epoch += 1
            validationAccuracy = dataset.get_validation_accuracy()
        return

        

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.numNeurons = 400

        self.wi = nn.Parameter(self.num_chars,self.numNeurons)
        self.bi = nn.Parameter(1,self.numNeurons)

        self.wr = nn.Parameter(self.numNeurons,self.numNeurons)
        self.br = nn.Parameter(1,self.numNeurons)

        self.wr2 = nn.Parameter(self.numNeurons,self.numNeurons)
        self.br2 = nn.Parameter(1,self.numNeurons)
        
        self.wf = nn.Parameter(self.numNeurons,5)
        self.bf = nn.Parameter(1,5)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        z = nn.Linear(xs[0],self.wi)
        zb = nn.AddBias(z,self.bi)
        h = nn.ReLU(zb)
        for i in range(1,len(xs)):
            zx = nn.Linear(xs[i], self.wi)
            zxb = nn.AddBias(zx,self.bi)
            zh = nn.Linear(h, self.wr)
            zhb =  nn.AddBias(zh,self.br)
            z = nn.Add(zxb,zhb)

            h1 = nn.ReLU(z)

            z2 = nn.Linear(h1,self.wr2)
            z2b = nn.AddBias(z2,self.br2)
            h = nn.ReLU(z2b)

            # self.wr2 = nn.Parameter(self.numNeurons,self.numNeurons)
            # self.br2 = nn.Parameter(1,self.numNeurons)
            # z = nn.Add(nn.Linear(xs[i], self.wi), nn.Linear(h, self.wr))
            # h = nn.ReLU(z)
        
        sum = nn.Linear(h,self.wf)
        predicted_y = nn.AddBias(sum,self.bf)
        return predicted_y


    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(xs)
        loss = nn.SoftmaxLoss(predicted_y,y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batchsize = 50
        learningRate = -1 * 0.05
        validationAccuracy = 0
        while validationAccuracy < 0.85:
            for x, y in dataset.iterate_once(batchsize):
                loss = self.get_loss(x,y)
                gwi,gbi,gwr,gbr,gwr2,gbr2,gwf,gbf = nn.gradients(loss,[self.wi,self.bi,self.wr,self.br,self.wr2,self.br2,self.wf,self.bf])
                
                self.wi.update(gwi,learningRate)
                self.bi.update(gbi,learningRate)

                self.wr.update(gwr,learningRate)
                self.br.update(gbr,learningRate)

                self.wr2.update(gwr2,learningRate)
                self.br2.update(gbr2,learningRate)

                self.wf.update(gwf,learningRate)
                self.bf.update(gbf,learningRate)

            validationAccuracy = dataset.get_validation_accuracy()
        return

