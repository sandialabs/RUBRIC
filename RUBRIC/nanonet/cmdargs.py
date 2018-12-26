import argparse
import os
import multiprocessing


class FileExist(argparse.Action):
    """Check if the input file exist."""
    def __call__(self, parser, namespace, values, option_string=None):
        if values is not None and not os.path.exists(values):
             raise RuntimeError("File/path for '{}' does not exist, {}".format(self.dest, values))
        setattr(namespace, self.dest, values)


class CheckCPU(argparse.Action):
    """Make sure people do not overload the machine"""
    def __call__(self, parser, namespace, values, option_string=None):
        num_cpu = multiprocessing.cpu_count()
        if int(values) <= 0 or int(values) > num_cpu:
            raise RuntimeError('Number of jobs can only be in the range of {} and {}'.format(1, num_cpu))
        setattr(namespace, self.dest, values)


class AutoBool(argparse.Action):
    def __init__(self, option_strings, dest, default=None, required=False, help=None):
        """Automagically create --foo / --no-foo argument pairs"""

        if default is None:
            raise ValueError('You must provide a default with AutoBool action')
        if len(option_strings)!=1:
            raise ValueError('Only single argument is allowed with AutoBool action')
        opt = option_strings[0]
        if not opt.startswith('--'):
            raise ValueError('AutoBool arguments must be prefixed with --')

        opt = opt[2:]
        opts = ['--' + opt, '--no-' + opt]
        if default:
            default_opt = opts[0]
        else:
            default_opt = opts[1]
        super(AutoBool, self).__init__(opts, dest, nargs=0, const=None,
                                       default=default, required=required,
                                       help='{} (Default: {})'.format(help, default_opt))
    def __call__(self, parser, namespace, values, option_strings=None):
        if option_strings.startswith('--no-'):
            setattr(namespace, self.dest, False)
        else:
            setattr(namespace, self.dest, True)

class ParseTransitions(argparse.Action):
    """Handle list of exactly 3 values, check values can be coerced to float and
    normalise so that they sum to 1.
    """
    def __init__(self, **kwdargs):
        kwdargs['metavar'] = ('stay', 'step', 'skip')
        super(ParseTransitions, self).__init__(**kwdargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # locally import these
        import numpy as np
        from dragonet.util.assertions import checkTransitionProbabilities
        try:
            values = np.array(values, dtype='float')
        except:
            raise ValueError('Illegal value for {} ({})'.format(option_string, values))
        values = values / np.sum(values)
        checkTransitionProbabilities(values)
        setattr(namespace, self.dest, values)


class ParseToNamedTuple(argparse.Action):
    """Parse to a namedtuple
    """
    def __init__(self, **kwdargs):
        assert 'metavar' in kwdargs, "Argument 'metavar' must be defined"
        assert 'type' in kwdargs, "Argument 'type' must be defined"
        assert len(kwdargs['metavar']) == kwdargs['nargs'], 'Number of arguments and descriptions inconstistent'
        assert len(kwdargs['type']) == kwdargs['nargs'], 'Number of arguments and types inconstistent'
        self._types = kwdargs['type']
        kwdargs['type'] = str
        self.Values = namedtuple('Values', ' '.join(kwdargs['metavar']))
        super(ParseToNamedTuple, self).__init__(**kwdargs)
        self.default = self.Values(*self.default) if self.default is not None else None

    def __call__(self, parser,  namespace, values, option_string=None):
        value_dict = self.Values(*[ f(v) for f, v in zip(self._types, values)])
        setattr(namespace, self.dest, value_dict)

def TypeOrNone(mytype):
    """Create an argparse argument type that accepts either given type or 'None'

    :param mytype: Type function for type to accept, e.g. `int` or `float`
    """
    def f(y):
        try:
            if y == 'None':
                res = None
            else:
                res = mytype(y)
        except:
            raise argparse.ArgumentTypeError('Argument must be None or {}'.format(mytype))
        return res
    return f


def NonNegative(mytype):
    """Create an argparse argument type that accepts only non-negative values

    :param mytype: Type function for type to accept, e.g. `int` or `float`
    """
    def f(y):
        yt = mytype(y)
        if yt < 0:
            raise argparse.ArgumentTypeError('Argument must be non-negative')
        return yt
    return f


def Positive(mytype):
    """Create an argparse argument type that accepts only positive values

    :param mytype: Type function for type to accept, e.g. `int` or `float`
    """
    def f(y):
        yt = mytype(y)
        if yt <= 0:
            raise argparse.ArgumentTypeError('Argument must be positive')
        return yt
    return f


def Vector(mytype):
    """Return an argparse.Action that will convert a list of values into a numpy
    array of given type
    """

    class MyNumpyAction(argparse.Action):
        """Parse a list of values into numpy array"""
        def __call__(self, parser, namespace, values, option_string=None):
            import tang.numpty as np
            try:
                setattr(namespace, self.dest, np.array(values, dtype=mytype))
            except:
                raise argparse.ArgumentTypeError('Cannot convert {} to array of {}'.format(values, mytype))
        @staticmethod
        def value_as_string(value):
            return ' '.join(str(x) for x in value)
    return MyNumpyAction

