from keras import backend as K
from keras.activations import relu

def make_ocnn_obj(nu, r):
    """`nu` is a hyperparameter that determines the cutoff for
    outliers.
    `r` is a model parameter that will be optimized after optimal
    weights are found for the fixed `r` given.
    """

    def ocnn_obj(_, y_pred):
        """`decision_score` is positive for ~normal~ points and negative
        for novel points. Objective function from
        https://arxiv.org/pdf/1802.06360.pdf
        """
        # the l2 norms of the weights are already added to the loss
        return (1 / nu) * K.mean(relu(r - y_pred))
