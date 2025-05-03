
class NamedTuple(object):
    __slots__ = ('score','state','parent','action','batch_size','device')

    def __init__(self, score, state, parent, action, batch_size, device):
        self.score      = score
        self.state      = state
        self.parent     = parent
        self.action     = action
        self.batch_size = batch_size
        self.device     = device

from typing import Any, Optional

class RecordMixin:
    """
    Provides NamedTuple-like helpers: _asdict, _replace, iteration…
    You can also put any shared properties/methods here.
    """
    def _asdict(self):
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}

    def _replace(self, **kwargs):
        return type(self)(**{**self._asdict(), **kwargs})

    def __iter__(self):
        for f in self.__dataclass_fields__.values():
            yield getattr(self, f.name)