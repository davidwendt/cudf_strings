import pynicucategory

def from_ndarray(arr, nulls=None):
    rtn = pynicucategory.n_createCategoryFromBuffer(arr, nulls)
    if rtn is not None:
        rtn = cucategory(rtn)
    return rtn

class cucategory:
    """
    """
    m_cptr = 0

    def __init__(self, cptr):
        self.m_cptr = cptr
        self._own = True

    def __del__(self):
        if self._own:
            pynicucategory.n_destroyCategory(self.m_cptr)
        self.m_cptr = 0

    def __str__(self):
        return str(self.keys())

    def __repr__(self):
        return "<cucategory[{}] keys={},values={}>".format(
            self.keys_type(), self.keys_size(), self.size())

    def get_cpointer(self):
        return self.m_cptr

    def size(self):
        return pynicucategory.n_size(self.m_cptr)

    def keys_size(self):
        return pynicucategory.n_keys_size(self.m_cptr)

    def keys(self, narr):
        pynicucategory.n_get_keys(self.m_cptr,narr)

    def keys_cpointer(self):
        return pynicucategory.n_keys_cpointer(self.m_cptr)

    def keys_type(self):
        return pynicucategory.n_keys_type(self.m_cptr)

    def values(self, narr):
        pynicucategory.n_get_values(self.m_cptr,narr)

    def values_cpointer(self):
        return pynicucategory.n_values_cpointer(self.m_cptr)

    def indexes_for_key(self, key, narr):
        return pynicucategory.n_get_indexes_for_key(self.m_cptr, key, narr)

    def to_type(self, narr, nulls=None):
        pynicucategory.n_to_type(self.m_cptr, narr, nulls)

    def gather_type(self, narr, nulls=None):
        pynicucategory.n_gather_type(self.m_cptr, narr, nulls)

    def gather_and_remap(self, indexes):
        rtn = pynicucategory.n_gather_and_remap(self.m_cptr, indexes)
        if rtn is not None:
            rtn = cucategory(rtn)
        return rtn

    def gather(self, indexes):
        rtn = pynicucategory.n_gather(self.m_cptr, indexes)
        if rtn is not None:
            rtn = cucategory(rtn)
        return rtn

    def gather_values(self, indexes):
        rtn = pynicucategory.n_gather_values(self.m_cptr, indexes)
        if rtn is not None:
            rtn = cucategory(rtn)
        return rtn

    def merge(self, cat):
        rtn = pynicucategory.n_merge_category(self.m_cptr, cat.m_cptr)
        if rtn is not None:
            rtn = cucategory(rtn)
        return rtn

    def add_keys(self, narr, nulls=None):
        rtn = pynicucategory.n_add_keys(self.m_cptr, narr, nulls)
        if rtn is not None:
            rtn = cucategory(rtn)
        return rtn

    def remove_keys(self, narr, nulls=None):
        rtn = pynicucategory.n_remove_keys(self.m_cptr, narr, nulls)
        if rtn is not None:
            rtn = cucategory(rtn)
        return rtn

    def remove_unused_keys(self):
        rtn = pynicucategory.n_remove_unused(self.m_cptr)
        if rtn is not None:
            rtn = cucategory(rtn)
        return rtn

    def set_keys(self, narr, nulls=None):
        rtn = pynicucategory.n_set_keys(self.m_cptr, narr, nulls)
        if rtn is not None:
            rtn = cucategory(rtn)
        return rtn
