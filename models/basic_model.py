'''

Todo: Add Documentation
Brief: This is the Parent Class for all other Models

'''

from abc import ABC, abstractmethod

class Basic_Model(ABC):

        @abstractmethod
        def create_model(self, x, weights, biases):
            pass

        @abstractmethod
        def format_data(self):
            pass

        @abstractmethod
        def run(self):
            pass
