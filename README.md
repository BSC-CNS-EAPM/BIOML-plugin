# BIOML-plugin

To install the plugin create a conda environment with the name bioml 
Go to the plugin folder BioML-plugin/Bioml/Include
Edit the utils.py in the line 438
```python
    # local
    elif cluster == "local":
        print("Generating local jobs...")
        print("Jobs", jobs)
        jobs = "conda run -n bioml " + jobs

    # change it to 
    jobs = "absolute_path_to_conda/conda run -n bioml (environment name) " + jobs
```

Use python 3.10
then install the packages needed for bioml by going to https://github.com/etiur/BioML.git and installing this package in the plugin environment
The pyproject.toml has all the necessary dependencies -> and then you also need 2 additional dependencies (mmseqs and perl-bio-featureio which can only be installed via conda)