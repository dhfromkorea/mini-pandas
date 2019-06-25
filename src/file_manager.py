import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(file_path, os.pardir))

from core import DataFrame


class FileManager(object):
    """Docstring for Reader. """

    def read_csv(self, file_path, delimiter, output_format="dataframe"):
        """TODO: to be defined1.

        @todo: for now, assumes header exists

        Parameters
        ----------
        file_path : TODO
        delimiter : TODO
        output_format : TODO, optional


        """
        import csv

        try:
            with open(file_path) as f:
                data = csv.reader(f, delimiter=delimiter)
                df = DataFrame(data)
        except Exception as e:
            raise Exception("Oops, invalid Input Data.\n{}".format(e))

        return df



    def to_csv(self, dataframe, file_path, include_header=True):
        """TODO: to be defined1.

        @todo: for now, assumes header exists

        Parameters
        ----------
        file_path : TODO
        delimiter : TODO
        output_format : TODO, optional


        """
        import csv
        try:
            with open(file_path, mode="w") as f:
                df_writer = csv.writer(f, delimiter=",", quotechar='"',
                        quoting=csv.QUOTE_MINIMAL)
                if include_header:
                    df_writer.writerow(dataframe.columns)

                for row in dataframe.iterrows():
                    assert isinstance(row, list)
                    row = [int(item) if isinstance(item, float) else item for item in row]
                    df_writer.writerow(row)

            #import pdb;pdb.set_trace()



        except Exception as e:
            raise Exception("Oops, invalid Input Data.\n{}".format(e))




