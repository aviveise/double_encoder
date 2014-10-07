__author__ = 'aviv'

from transformer_base import TransformerBase

class IdentityTransformer(TransformerBase):

    def __init__(self):
        super(IdentityTransformer, self).__init__(None)

    def compute_outputs(self, test_set_x, test_set_y):

        return test_set_x, test_set_y



