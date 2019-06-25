# A minimalistic reproduction of key Pandas API endpoints

The aim is to develop a small library that supports a set of common features found in Pandas.

## Getting started

```
chmod +x run.sh
./run.sh
```

## Supported endpoints
- [x] DataFrame
-   [x] indexing by integers, slices, list
-   [x] indexing by column names
-   [x] df.shape, df.columns, df.dtypes
-   [x] df.merge(left, right, left_on=c1, right_on=c2)
-   [x] df.groupby(by=col_name)
-   [x] df.sort_values(by=[c1,c2], ascending=[0,1])
-   [x] df.is_na()
-   [x] df.fill_na(value)
-   [x] df.replace(pattern, value)
-   [x] df.dropna()
-   [x] df.drop_duplicate(subset=[c1,c2])
- [x] GroupBy
-   [x] g.sum()
-   [x] g.map(func)
-   [x] g.reduce(func)
-   [x] g.filter(func)
-   [x] g.count()

## Examples
In MiniPandas, we can do things like:

### Example-1

For instance, you can enjoy the API endpoints as you would with pandas.
```python
# indexing
empty_row = df[:0]
single_row = df[3]
indexing_by_str_columns = df[:, ["a", "c"]]
indexing_by_int_columns = df[:, [0, 3]]

# groupby and map/reduce/filter
groups = df.groupby(by=[3])
query_1 = lambda x : x * 2
res1 = groups.map(query_1)

# cleansing
df.is_na()
df_no_na = df.fill_na("hello_world")
df_no_empty_str = df.replace("", float("nan"))
df_cleansed = df.drop_na()

# count, merge, sort_values
g_uniq = df_cleansed_uniq.groupby(by=['Rooms'])
df_num_rooms = g_uniq.count()
df_num_rooms = df_num_rooms.rename({"Acres": "Acers"})
df_joined = df_num_rooms.merge(df_total_num_rooms, left_on=["Age"],
        right_on=["Taxes"])
df_output = df_joined.sort_values(by=["Taxes", "Age"], ascending=[0, 1])
```

### Example-2 

```python
# read from csv to dataframe
fm = FileManager()
df = fm.read_csv(input_path, delimiter=",", output_format="dataframe")


if df.is_empty:
    df_output = df
    df_output.columns = ["Age", "Taxes", "Acres"]
    fm.to_csv(df_output, output_path)
    return


# data cleansing
df_no_empty_str = df.replace("", float("nan"))
df_cleansed = df_no_empty_str.drop_na()

# data deduplicate
cols_to_be_unique = ['Living','Taxes', 'Acres']
df_cleansed_uniq = df_cleansed.drop_duplicates(subset=cols_to_be_unique)


g = df_cleansed.groupby(by=['Living'])
g.groups
df = g.sum()
df_room = df_room[:, ["Living", "LivingRoom"]]
df_room = df_room.rename({"Taxes": "PropertyTaxes"})

g_uniq = df_cleansed_uniq.groupby(by=['Room'])
df_taxes = g_uniq.count()
df_taxes = df_taxes.rename({"Room": "NumberOfRooms"})

df_joined = df_taxes.merge(df_room, left_on=["NumberOfRooms"],
        right_on=["LivingRoom"]) 

N, M = df_joined.shape
if N > 1:
    # sort the dataframe by columns
    df_output = df_joined.sort_values(by=["NumberOfRooms", "Taxes"], ascending=[0, 1])
else:
    df_output = df_joined

# write the dataframe to csv
fm.to_csv(df_output, output_path)
```

### Prerequisites
* python 3.6+

