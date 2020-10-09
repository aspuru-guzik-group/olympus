#!/usr/bin/env python


from olympus import Logger

from .abstract_object import Object


class Config(Object):

    def __init__(self, from_dict=None, from_json=None, name='CustomConfig'):
        """

        Args:
            from_dict:
            from_json:
            name:
        """
        super(Config, self).__init__(me=name)
        self.name = name

        if from_dict is not None:
            self.from_dict(info_dict=from_dict)
        elif from_json is not None:
            self.from_json(json_file=from_json)

        if from_dict is not None and from_json is not None:
            message = 'you have passed both "from_dict" and "from_json" arguments to Config: "from_json" will be discarded'
            Logger.log(message, 'WARNING')

    def __repr__(self):
        return f"<Config (name={self.me})>"

    def __str__(self):
        # determine spacing - additional numbers is to ensure a minimum spacing due to the headers
        max_prop_len = max([self.max_prop_len, 9])
        max_attr_len = max([len(str(att)) for att in self.attrs] + [5])
        max_type_len = max([len(str(type(att).__name__)) for att in self.attrs] + [4])
        total = max_prop_len + max_attr_len + max_type_len + 6

        # Title
        string = '=' * total + '\n'
        ind = int(round((total - len(self.me) - 8) / 2, 0))
        string += ' ' * ind + f'Config: {self.me}\n'
        string += '=' * total + '\n'

        # Headers
        string += '{0:<{3}}{1:<{4}}{2:<{5}}\n'.format('Parameter', 'Value', 'Type',
                                                      max_prop_len + 2,
                                                      max_attr_len + 2,
                                                      max_type_len + 2)
        string += '-' * total + '\n'

        # Parameters/Arguments
        for prop in sorted(self.props):
            string += '{0:<{3}}{1:<{4}}{2:<{5}}\n'.format(prop,
                                                          str(getattr(self, prop)),
                                                          str(type(getattr(self, prop)).__name__),
                                                          max_prop_len + 2,
                                                          max_attr_len + 2,
                                                          max_type_len + 2)
        string += '=' * total + '\n'
        return string
