from abc import ABCMeta, abstractmethod

class PositionFactory:
    __metaclass__ = ABCMeta

    factory_dict = {}

    @staticmethod
    def set_factory(type, factory):
        PositionFactory.factory_dict[type] = factory

    @staticmethod
    def factory(type='pony'):
        return PositionFactory.factory_dict[type]

    @abstractmethod
    def create(self, n=0, state=None):
        pass

class ModelFactory:
    __metaclass__ = ABCMeta

    factory_dict = {}

    @staticmethod
    def set_factory(type, factory):
        ModelFactory.factory_dict[type] = factory

    @staticmethod
    def factory(type='recurrent'):
        return ModelFactory.factory_dict[type]

    @abstractmethod
    def create(self):
        pass
