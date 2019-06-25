import os
import argparse


file_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(file_path, os.path.pardir))

from file_manager import FileManager

def argsparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str,
            help="relative to input folder", default="input.txt")
    parser.add_argument('--output_path', type=str,
    help="relative to output folder", default="output.txt")
    return parser.parse_args()


def main(args):
    input_path = os.path.join(root_path, "input", args.input_path)
    output_path = os.path.join(root_path, "output", args.output_path)

    # read from csv to dataframe
    fm = FileManager()
    df = fm.read_csv(input_path, delimiter=",", output_format="dataframe")


    if df.is_empty:
        df_output = df
        df_output.columns = [DEFAULT_COLUMNS]
        fm.to_csv(df_output, output_path)
        return


    # data cleansing
    df_no_empty_str = df.replace("", float("nan"))
    df_cleansed = df_no_empty_str.drop_na()

    # data deduplicate
    cols_to_be_unique = [DEFAULT_UNIQUE_COLUMNS]
    df_cleansed_uniq = df_cleansed.drop_duplicates(subset=cols_to_be_unique)


    # do your work

    # write the dataframe to csv
    fm.to_csv(df_output, output_path)


if __name__ == "__main__":
    args = argsparser()
    main(args)
