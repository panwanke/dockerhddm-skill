# lan add new models to hddm tutorial UNUSED ATM

> Converted from `hddm/hddm/examples/lan_add_new_models_to_hddm_tutorial_UNUSED_ATM.ipynb`. Code is preserved; rich outputs are omitted.

## HOW TO ADD A NEW MODEL TO HDDM

This (living) document shows you how to add your own model with LAN likelihood to your HDDM fork.
We are working on a version that will allow you to supply custom likelihoods unrelated to the LAN framework, but for now.. let's get to it.

There are *four* components to including a full LAN based model into HDDM. 

1. A *config dictionary* that specifies metadata concerning your model

2. The *LAN* or another *callable likelihood function*
   
3. A *likelihood function* that is PyMC2 compatible and makes use of your LAN
   
4. A *simulator* that allows you to generate data from your model

To begin with, we assume your model has a name: Let'c call it `test_model`

### 1. The CONFIG DICTIONARY

You find a central config dictionary (`model_config`) under `hddm/model_config.py`. 
For each model it stores a range of basic metadata (`dictionary` in `dictionary`).
The rest of the codebase draws upon the data in this `dictionary` repeatedly.

For a given model, let's take the `ddm` as an example, a metadata dictionary looks like this,

```python
"ddm": {
        "params": ["v", "a", "z", "t"], # parameter names associated to your model 
        "params_trans": [0, 0, 1, 0], # for each parameter do you want the sampler to transform it into an unconstrained space? (invlogit <--> logistic)
        "params_std_upper": [1.5, 1.0, None, 1.0], # for group models, what is the maximal standard deviation to consider for the prior on the parameter
        "param_bounds": [[-3.0, 0.3, 0.1, 1e-3], [3.0, 2.5, 0.9, 2.0]], # the parameter boundaries you used for training your LAN
        "boundary": hddm.simulators.bf.constant, # add a boundary function (if relevant to your model) (optional)
        "n_params": 4, # number of parameters of your model
        "params_default": [0.0, 1.0, 0.5, 1e-3], # defaults for each parameter 
        "hddm_include": ["z"], # suggestion for which parameters to include via the include statement of an HDDM model (usually you want all of the parameters from above)
        "n_choices": 2, # number of choice options of the model
        "choices": [-1, 1], # choice labels (what your simulator spits out)
        "slice_widths": {"v": 1.5, "v_std": 1,  
                         "a": 1, "a_std": 1, 
                         "z": 0.1, "z_trans": 0.2, 
                         "t": 0.01, "t_std": 0.15}, # hyperparameters for the slice-sampler used for posterior sampling, take these as an orientation, can be helpful to optimize speed (optional)
    }

```

Add the metadata dictionary for your model (`test_model`) to the `model_config` dictionary in `hddm/model_config.py`.

### 2. The LAN

We will assume that your are using Keras for now. 

If you are not using keras, make sure to wrap your model forward pass, so that your network has a `predict_on_batch()` method.

This function is used by the likelihood to batch process trial by trial data and receive log likelihoods.

You find the `load_model.py` file in the `hddm\keras_models` folder in the package. Add the following code to the `load_mlp()` function in that file:

```python     
if model == "test_model":
    return keras.models.load_model(hddm.keras_models.__path__[0] + \ 
           "/keras_model_file_name.h5", compile=False) 
```

### 3. Likelihood Function

We don't only need to supply the network, but in addition we need to define a function which wraps our network into a PyMC2 recognizable likelihood.
Using your config dictionary, you can auto-generate a string to define yourself such a likelihood function, using the following utility function.
You need to do this twice, once for standard HDDM style likelihoods, and once for regression based likelihoods.

```python
    my_likelihood_string = hddm.utils.make_likelihood_str_mlp(config = hddm.model_config['test_model'], fun_name = 'your_function_name')
    my_likelihood_string_reg = hddm.utils.make_reg_likelihood_str_mlp(config = hddm.model_config['test_model'], fun_name = 'your_function_name')
```

(The string defines a function `custom_likelihood`. You might want to rename it.)

You can now add these likelihood functions (just copy the string as code, taking care of linebreaks) to the file `likelihoods_mlp.py` in the `hddm` folder.

Do this once with your string `my_likelihood_string` and once with the string `my_likelihood_string_reg`, respectively adding these to the `make_mlp_likelihood()`, and `make_mlp_likelihood_reg()` functions
in the `likelihoods_mlp.py` file.

In the `make_mlp_likelihood()` function you find the definition of multiple such likelihoods.

Add yours at the end (just before the `likelihood_funs` dictionary is defined), then add it to the `likelihood_funs` dictionary, like so:

```python
    likelihood_funs['test_model'] = name_of_our_new_likelihood_function
```

This takes care of the function defined with `my_likelihood_string`. Adding `my_likelihood_string_reg` to `make_mlp_likelihood_reg()` works analogously.

### 4. Simulator (Optional)

To add a simulator, simply extend the `simulator()` function in the `hddm/simulators/basic_simulator.py` file.

Add a code snipped like this one to the function:

```python
if model == "test_model":
        x = test_model_simulator(
                parameter_1=theta[:, 0], # basic simulator parameters (relevant model paramters for fitting the model etc.)
                parameter_2=theta[:, 1], 
                parameter_3=theta[:, 2],
                ...
                s=s, # --> standard deviation of diffusion (optional)
                n_samples=n_samples, # --> number of samples to generate 
                n_trials=n_trials, # --> n_trials, see doc string for simulator()
                delta_t=delta_t, # --> time step size that the simulator uses (optional)
                boundary_params={}, # --> boundary parameters (in case the simulator takes in a boundary function) (optional)
                boundary_fun=bf.constant, # --> boundary function (optional)
                boundary_multiplicative=True, # --> whether the particular boundary is multiplicative or additive with respect to an initial boundary separation (optional)
                max_t=max_t, # --> time cutoff for the simulator (optional)
        )
```

Where `theta` is supplied as a `list` of values, or a `numpy.array` and the `test_model_simulator()` is your model simulator. 

Add your model simulator either as a separate python or *cython* file and import it at the beginning of the `basic_simulator.py` file. 

If you add a *cython* simulator, make sure to extend the `setup.py` file in the package folder, so that it compiles your simulator at installation of the package.

### Enjoy your HDDM-extension

If you went though steps 1-3 or 1-4 above, you should now reinstall your HDDM-fork and have your new `test_model` available.

WARNING: Not all plots will immediately work with your model, but usually the `hddm.plotting.caterpillar_plot()`, the `hddm.plotting.posterior_pair_plot()` and for 2-choice models the `hddm.plotting.posterior_predictive_plot()` should be operational.

If your LAN and model work fine, _please share your efforts with the community_ and turn your code additions into a `pull-request` on the main repository, [here](https://github.com/hddm-devs/hddm/tree/master/hddm).
