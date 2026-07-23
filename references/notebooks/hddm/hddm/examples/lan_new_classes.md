# lan new classes

> Converted from `hddm/hddm/examples/lan_new_classes.ipynb`. Code is preserved; rich outputs are omitted.

## New Classes

The **LAN extension (HDDM >= 0.9.0)**, provides three new classes which are LAN-enabled versions of the respective classes in base HDDM.
These new classes are, 

- The `HDDMnn()` class
- The `HDDMnnStimCoding()` class
- The `HDDMnnRegressor()` class

The usage mirrors what you are used to from standard `HDDM` equivalents. 

What changes is that you now use the `model` argument to specify one of the models you find listed in the `hddm.model_config.model_config` dictionary (you can also provide a custom model, for which you should look into the respective section in this documentation).

Moreover, you have to be a little more careful when specifying the `include` argument, since the ability to use new models comes with new parameters. To help get started here, the `hddm.model_config.model_config` dictionary provides you a `hddm_include` key for *every* model-specific sub-dictionary. This let's you fit all parameters of a given model. To keep some parameters fixed, remove them respectively from the resulting list.

### Short example

### Code cell 5

```python
import hddm
```

### Code cell 6

```python
model = "angle"
cavanagh_data = hddm.load_csv(hddm.__path__[0] + "/examples/cavanagh_theta_nn.csv")
model_ = hddm.HDDMnn(
    cavanagh_data,
    model=model,
    include=hddm.model_config.model_config[model]["hddm_include"],
    is_group_model=False,
)
```

### Code cell 7

```python
model_.sample(1000, burn=200)
```

### Code cell 8

```python
model_.get_traces()
```

### Code cell 9

```python
model_.gen_stats()
```
