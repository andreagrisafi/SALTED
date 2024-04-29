# ~ Test MKDocs (playground)

## Installation

To use SALTED, first install it using pip:

```console
git clone bulabula
python -m pip install -e .
```

## Get DF basis infomation

To get density fitting basis information,
you can use the `salted.get_basis_info.build()` function:

::: salted.get_basis_info.build

<br>


<!-- 
The `kind` parameter should be either `"meat"`, `"fish"`, or `"veggies"`.
Otherwise, [`get_random_ingredients`][lumache.get_random_ingredients] will raise an exception [`lumache.InvalidKindError`](/api#lumache.InvalidKindError). -->

For example:

```python
>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']
```