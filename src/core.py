from abc import ABC, abstractmethod, abstractproperty
import math

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(file_path, os.pardir))


class DataFrameBase(ABC):
    @property
    def shape(self):
        pass


    @property
    def columns(self):
        pass


    @property
    def dtypes(self):
        pass


    @property
    def shape(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass


    @abstractmethod
    def __getitem__(self, pos):
        pass


    @abstractmethod
    def as_matrix(self):
        pass


    @abstractmethod
    def groupby(self, by):
        pass


    @abstractmethod
    def sort_values(self, by, ascending):
        return


    @abstractmethod
    def is_na(self):
        return


    @abstractmethod
    def fill_na(self):
        return


    @abstractmethod
    def replace(self):
        return


    @abstractmethod
    def drop_na(self):
        return


    @abstractmethod
    def drop_duplicates(self, subset):
        return


    @abstractmethod
    def merge(self, right, left_on, right_on):
        return


    @abstractmethod
    def filter(self, func, axis=0):
        return


    @abstractmethod
    def rename(self, rename_map):
        pass

class DataFrame(DataFrameBase):
    """
    """
    DTYPE_INT = "int"
    DTYPE_FLOAT = "float"
    DTYPE_STR = "str"
    DTYPE_BOOL = "bool"


    def __init__(self, iterable, columns=None):
        """TODO: Docstring for __init__.

        Parameters
        ----------
        iterable : TODO

        Returns
        -------
        TODO

        """
        #DataFrameBase.__init__(self)

        data = {}
        self._is_empty = False

        has_columns_in_header = columns == None

        if has_columns_in_header:
            # assumesthe header included in the first row
            data["columns"] = next(iterable)
        else:
            data["columns"] = columns


        rows = [row for row in iterable]

        # compute dataframe shape
        # (N, M)
        if len(rows) == 0:
            # rows = [] -> [[]]
            data["rows"] = [rows]
            self._is_empty = True
            n_row = n_col = 0
        elif not isinstance(rows[0], list):
            # @todo: potentially abstract this out to Series
            # rows = [1,2,3...] -> [[1,2,3]]
            data["rows"] = [rows]
            n_row = 1
            n_col = len(rows)
        else:
            # rows = [[1,2], [3.4]]
            data["rows"] = rows
            n_row = len(rows)
            n_col = len(rows[0])


        # infer dtype
        if n_row > 0 and n_col > 0:
            data["dtypes"] = DataFrame.infer_dtype(data["rows"][0])
            # convert by dtypes
            # @todo: fix this repeat loop
            # O(2n)
            rows = []
            for row in data["rows"]:
                row_typed = DataFrame.coerce_type(row, data["dtypes"])
                rows.append(row_typed)

            data["rows"] = rows
        else:
            data["dtypes"] = None


        self._data = data
        self._columns = data["columns"]
        self._dtypes = data["dtypes"]
        self._shape = (n_row, n_col)


    @property
    def is_empty(self):
        return self._is_empty


    def as_matrix(self):
        """
        N > 1, M > 1 -> [[]]
        N = 1 or M = 1 -> []
        N = 0 or M = 0 -> []
        """
        N, M = self._shape
        if N > 1 and M > 1:
            return self._data["rows"]
        elif N == 1:
            # [[1,23,4,5]]
            return self._data["rows"][0]
        elif M == 1:
            # [[1], [2], [3]]
            return [row[0] for row in self._data["rows"]]
        else:
            # [[]]
            return []


    @property
    def dtypes(self):
        return self._dtypes


    @property
    def shape(self):
        return self._shape


    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, cols):
        self._columns = cols


    @classmethod
    def infer_dtype(cls, data):
        """
        infer dtype from the first row

        currently supporting:
        1. str, 2. int, 3.float (deprecated)

        now we support only

        1. str (non-numerical) 2. float (numerical)

        Parameters
        ----------
        cls :
        data : list of items

        Returns
        -------
        TODO

        """
        # store dtypes
        dtypes = [None] * len(data)
        for i, item in enumerate(data):
            try:
                _ = float(item)
                #if isinstance(item, int) or item.isdigit():
                #    # int
                #    dtypes[i] = cls.DTYPE_INT
                #else:
                    # float
                dtypes[i] = cls.DTYPE_FLOAT
            except ValueError:
                # str
                dtypes[i] = cls.DTYPE_STR

        return dtypes


    @classmethod
    def coerce_type(cls, data, dtypes):
        coerced = []

        for item, dtype in zip(data, dtypes):

            #if dtype == cls.DTYPE_INT:
            #    if item == "": # could NOT be str
            #        # @todo: super ugly. refactor later
            #        item = float('nan')
            #    else:
            #        item = int(item)
            if dtype == cls.DTYPE_FLOAT:

                if item == "": # could NOT be str
                    # @todo: super ugly. refactor later
                    item = float('nan')
                else:
                    item = float(item)
            elif dtype == cls.DTYPE_STR:
                pass
            elif dtype == cls.DTYPE_BOOL:
                if item == "": # could NOT be str
                    # @todo: super ugly. refactor later
                    item = float('nan')
                else:
                    item = bool(item)
            else:
                raise Exception("Unknown data type")
            coerced.append(item)

        return coerced



    def iterrows(self):
        for row in self._data["rows"]:
            yield row


    def itercols(self):
        for j in range(self._shape[1]):
            yield self[:, j].as_matrix()


    def __repr__(self):
        msg = "\n"
        if self._is_empty:
            msg += "Empty DataFrame"
            return msg

        msg += "    "
        for j, c in enumerate(self._columns):
            msg += "{}     ".format(j)

        msg += "\n"

        rows = self._data["rows"]
        for i, r in enumerate(rows):
            msg += "{}  ".format(i)
            for item in r:
                msg += "{} ".format(item)
            msg += "\n"

        msg += "DataFrame: {}\n".format(self._shape)
        msg += "Columns: {}\n".format(self._columns)
        msg += "dtypes: {}\n".format(self._dtypes)

        return msg


    def _select_data(self, data, pos):
        """TODO: Docstring for function.

        Parameters
        ----------
        data : [[]]
        pos : int, slice, [int]

        Returns
        -------
        [[]]

        """
        subdata = []
        # select rows
        try:
            if isinstance(pos, list):
                # [i,j]
                for i in pos:
                    # in case i is string
                    if isinstance(i, str) and i in self._columns:
                        j = self._columns.index(i)
                        row = data[j]

                    elif isinstance(i, int):
                        row = data[i]
                    else:
                        raise ValueError
                    subdata.append(row)

            elif isinstance(pos, slice):
                # i:j
                subdata = data[pos]
            elif isinstance(pos, int):
                # int
                row = data[pos]
                subdata.append(row)
            else:
                raise IndexError
        except IndexError:
            import pdb;pdb.set_trace()

        return subdata


    def __getitem__(self, pos):
        if isinstance(pos, tuple):
            # select rows and columns
            rows = self._select_data(self._data["rows"], pos[0])
            data = []
            for r in rows:
                c = self._select_data(r, pos[1])
                data.append(c)

            if isinstance(pos[1], list):
                cols = []
                for c in pos[1]:
                    # in case c is string
                    if isinstance(c, str) and c in self._columns:
                        j = self._columns.index(c)
                        col = self._columns[j]
                    elif isinstance(c, int):
                        col = self._columns[c]
                    else:
                        raise ValueError

                    cols.append(col)
            elif isinstance(pos[1], slice):
                cols = self._columns[pos[1]]
            elif isinstance(pos[1], int):
                cols = [self._columns[pos[1]]]
            else:
                raise IndexError("index error")
        else:
            data = self._select_data(self._data["rows"], pos)
            cols = self._columns


        if len(data) == 1:
            # shave off the outer []
            data = data[0]

        """
        data can be a_{ij}, [], [[]]
        """
        return DataFrame(data, cols)


    def _filter(self, func, axis=0):
        N, M = self._shape
        indices = []

        if axis == 0:
            for j in range(M):
                col = self[:, j].as_matrix()
                if func(col):
                    indices.append(j)

            return self[:, indices]

        elif axis ==1:
            for i in range(N):
                row = self[i, :]
            if func(row):
                indices.append(j)

            return self[indices,:]

        else:
            raise NotImplemented


    def filter(self, func, axis=0):
        """
        axis=0, by column
        axis=1, by row
        return columns where func(column) true

        Parameters
        ----------
        func : TODO
        axis : TODO, optional

        Returns
        -------
        return DataFrame

        """
        return self._filter(func, axis)


    def groupby(self, by):
        return GroupBy(dataframe=self, by=by)


    def is_na(self):
        """

        nan or empty

        @note: potential performance degradation: float casting

        Parameters
        ----------

        Returns
        -------
        return DataFrame (bool)

        """

        rows_is_na = []


        for j, col in enumerate(self.itercols()):
            # check for "" or nan
            if self._dtypes[j] == DataFrame.DTYPE_STR:
                col_is_na = list(map(lambda x : x == "", col))
            else:
                # numeric columns
                # @note: make a numeric typecheck method
                # @note: includes boolean?
                col_float = list(map(float, col))
                col_is_na = list(map(math.isnan, col_float))


            # store the results
            if len(rows_is_na) == 0:
                for _ in range(self._shape[0]):
                    rows_is_na.append([])

            for i, item in enumerate(col_is_na):
                # fill a_i,j for all i's
                rows_is_na[i].append(item)


        # @todo: remove this
        for i, _ in enumerate(rows_is_na):
            for item in rows_is_na[i]:
                assert isinstance(item, bool)

        return DataFrame(rows_is_na, columns=self._columns)


    def fill_na(self, value):
        raise NotImplementedError


    def drop_na(self):
        """
        not an in-place method!!
        """
        from functools import reduce
        rows = []
        # check for nan
        # https://stackoverflow.com/questions/944700/how-can-i-check-for-nan-values
        is_not_nan = lambda x, y : x  * (y == y)

        for row in self.iterrows():
            if reduce(is_not_nan, row, 1):
                # don't have to drop
                rows.append(row)

        return DataFrame(rows, columns=self._columns)


    def _cols_to_int_indices(self, col_list):
        """
        convert columns (str or int)
        to columns (int)
        """
        cols_int = []
        for col in col_list:
            if isinstance(col, int):
                cols_int.append(col)
            elif col in self._columns:
                j = self._columns.index(col)
                cols_int.append(j)
            else:
                raise ValueError

        return cols_int


    def _cols_to_str_names(self, col_list):
        """
        convert columns (str or int)
        to columns (int)
        """
        cols_str = []
        for col in col_list:
            if col in self._columns:
                cols_str.append(col)
            elif isinstance(col, int):
                cols_str.append(self._columns[col])
            else:
                raise ValueError

        return cols_str


    def drop_duplicates(self, subset):
        """

        """
        assert isinstance(subset, list)
        cols = self._cols_to_int_indices(subset)

        rows_dedup = []
        ht = {}
        for row in self.iterrows():
            key = []
            for j in cols:
                key.append(row[j])

            if not tuple(key) in ht:
                ht[tuple(key)] = True
                rows_dedup.append(row)

        return DataFrame(rows_dedup, columns=self._columns)


    def replace(self, v_from, v_to):
        """
        not an in-place method!!
        """
        replaced_list = []
        for row in self.iterrows():
            replaced = [v_to if e == v_from else e for e in row]
            replaced_list.append(replaced)
        return DataFrame(replaced_list, columns=self._columns)


    def merge(self, right, left_on, right_on):
        """
        left : DataFrame
        right : DataFrame

        assumes only one shared column
        """
        return self._merge(right, left_on, right_on)


    def _merge(self, right, left_on, right_on):
        """
        create a row
        concatenating key c_l's c_r's
        duplicates allowed
        """
        left = self
        assert isinstance(left_on, list)
        assert isinstance(right_on, list)
        groups_l = left.groupby(by=left_on).groups
        groups_r = right.groupby(by=right_on).groups
        col_c_l = [groups_l["metadata"][k] for k in groups_l["metadata"]][0]
        col_c_r = [groups_r["metadata"][k] for k in groups_r["metadata"]][0]

        new_rows = []
        for key in groups_l["data"]:
            # left key shared
            row_indices_l = groups_l["data"][key]
            row_indices_r = groups_r["data"][key]
            # produce cartesian product
            key_vals = [k for k in key]
            for i_l in row_indices_l:
                for i_r in row_indices_r:
                    row_l = left[i_l, col_c_l].as_matrix()
                    row_r = right[i_r, col_c_r].as_matrix()
                    new_row = key_vals + row_l + row_r
                    new_rows.append(new_row)


        for key_indices in groups_l["metadata"]:
            # int_cols
            col_indices = groups_l["metadata"][key_indices]
            cols = list(key_indices) + col_indices
        cols_l = left._cols_to_str_names(cols)

        for key_indices in groups_r["metadata"]:
            # int_cols
            col_indices = groups_r["metadata"][key_indices]
            cols = col_indices
        cols_r = right._cols_to_str_names(cols)
        new_cols = cols_l + cols_r


        return DataFrame(new_rows, columns=new_cols)


    def rename(self, rename_map):
        return self._rename(rename_map)


    def _rename(self, rename_map):
        cols = self._columns
        for c_old in rename_map:
            j = cols.index(c_old)
            if j == -1:
                raise ValueError
            cols[j] = rename_map[c_old]
        return self


    def sort_values(self, by, ascending):
        """
        """
        assert isinstance(by, list)
        assert isinstance(ascending, list)
        criteria = self._cols_to_int_indices(by)

        rows = self.as_matrix()

        # @note: potentiall slow
        # assuming stable sort
        for c, asc in zip(reversed(criteria), reversed(ascending)):
            if asc:
                rows = sorted(rows, key=lambda x : x[c])
            else:
                rows = sorted(rows, key=lambda x : x[c], reverse=True)

        return DataFrame(rows, self._columns)

class GroupByBase(ABC):
    """Docstring for GroupBy. """

    @property
    def groups(self):
        pass


    @abstractmethod
    def __repr__(self):
        pass


    @abstractmethod
    def reduce(self, func):
        pass


    @abstractmethod
    def map(self, func):
        pass


    @abstractmethod
    def to_dataframe(self):
        pass


    @abstractmethod
    def count(self):
        pass

class GroupBy(GroupByBase):
    """
    aims to support
    - splitting
    - reduce
    - map
    - filter

    GroupBy object is a map
    (grouped column values)
    -> [row index]


    """

    def __init__(self, dataframe, by):
        """
        expects by to be integers
        """

        assert isinstance(dataframe, DataFrame)
        assert isinstance(by, list)

        self._df = dataframe
        self._cols = cols = dataframe._cols_to_int_indices(by)

        cols_c = set(range(len(self._df._columns))) - set(self._cols)
        self._cols_c = sorted(list(cols_c))


        groups = {}
        groups["metadata"] = {tuple(self._cols): self._cols_c}
        groups["data"] = {}

        for i, row in enumerate(dataframe.iterrows()):
            key = len(cols) * []
            for col in cols:
                key.append(row[col])

            key = tuple(key)
            if key in groups["data"]:
                groups["data"][key].append(i)
            else:
                groups["data"][key] = [i]

        self._groups = groups



    @property
    def groups(self):
        """
        group is a map
        from key column values (tuple)
        to row indices (list)

        """
        return self._groups


    def __repr__(self):
        msg = "\n"

        for key in self._groups["data"]:
            msg += "{} -> ".format(key)
            row_indices = self._groups["data"][key]
            msg += "{}\n".format(row_indices)

        return msg



    def reduce(self, func):
        return self._reduce(func)


    def _reduce(self, func):
        """
        for reduce types of operations
        e.g. sum

        Parameters
        ----------
        func : TODO

        Returns
        -------
        TODO

        """


        def check_numeric_column(col):
            """
            gets a Series or list
            returns True if of int/float
            """
            dtype = DataFrame.infer_dtype(col)[0]
            #is_int = dtype == DataFrame.DTYPE_INT
            is_float = dtype == DataFrame.DTYPE_FLOAT
            return is_float


        from functools import reduce
        groups_reduced = {}

        groups_reduced["metadata"] = {tuple(self._cols): self._cols_c}
        groups_reduced["data"] = {}


        groups = self._groups["data"]
        for key in groups:
            row_indices = groups[key]
            group = self._df[row_indices]
            # group is a sub-dataframe
            #group = group._filter(check_numeric_column)

            #columns_reduced = [None] * group.shape[1]
            # exclude key columns (by)
            columns_reduced = []


            for j in self._cols_c:
                #assert isinstance(group[:, j].as_matrix(), list)
                res = reduce(func, group[:, j].as_matrix())
                columns_reduced.append(res)

            groups_reduced["data"][key] = columns_reduced

        return groups_reduced


    def map(self, func):
        """

        currently data structure is represented in rows
        hence, inefficient.

        @todo: fix this

        Returns
        -------
        DataFrame of every element <- func(element)

        """
        return self._map(func)


    def _map(self, func):
        res = []
        for row in self._df.iterrows():
            applied = list(map(func, row))
            res.append(applied)

        return DataFrame(res, self._df.columns)


    def sum(self):
        """

        exclude by_columns from applying
        should return dataframe

        currently data structure is represented in rows
        hence, inefficient.

        @todo: fix this

        Parameters
        ----------
        func : TODO

        Returns
        -------

        """
        for key_indices in self._groups["metadata"]:
            # int_cols
            col_indices = self._groups["metadata"][key_indices]
            cols = list(key_indices) + col_indices
        cols = self._df._cols_to_str_names(cols)

        groups = self._reduce(lambda x, y : x + y)["data"]
        rows = []
        for key in groups:
            row = [k for k in key]
            row += groups[key]
            rows.append(row)

        return DataFrame(rows, columns=cols)


    def to_dataframe(self):
        return self._to_dataframe()


    def _to_dataframe(self):
        """
        GroupBy (map) to DataFrame
        """
        assert isinstance(groups, dict)

        for key_indices in self._groups["metadata"]:
            # int_cols
            col_indices = self._groups["metadata"][key_indices]
            cols = list(key_indices) + col_indices
        cols = self._df._cols_to_str_names(cols)


        rows = []
        for key in groups["data"]:
            # value
            row = [k for k in key]
            row_indices = groups["data"][key]
            for i in row_indices:
                l = self._df[i, col_indices].as_matrix()
                assert isinstance(l, list)
                row += l
            #df = self._df[row_indices, col_indices]
            rows.append(row)

        return DataFrame(rows, columns=cols)


    def count(self):
        return self._count()


    def _count(self):

        for key_indices in self._groups["metadata"]:
            # int_cols
            # @note: hardcode
            key_indices = self._df._cols_to_str_names(list(key_indices))
            col_count = "_".join(key_indices + ["count"])
            cols = key_indices + [col_count]


        new_groups = {}
        for key in self._groups["data"]:
            new_groups[key] = len(self._groups["data"][key])

        rows = []
        for key in new_groups:
            row = [k for k in key]
            row.append(new_groups[key])
            rows.append(row)


        return DataFrame(rows, columns=cols)


