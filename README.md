Code for the TMLR paper [A Variational Perspective on Generative Flow Networks](https://openreview.net/forum?id=AZ4GobeSLq&invitationId=TMLR/Paper612/) by Heiko Zimmermann, Fredrik Lindsten,  Jan-Willem van de Meent, and Christian A Naesseth.

**Requirements:**
All required packages are listed in `req.pip` and can be installed by running `pip install -r req.pip`.

**Running experiments:**
- Clone repository and run `git submodule init` followed by `git submodule update` to initiallize and fetch the nesseccary submodule. The submodule is needed for the synthetic density experiments only and is a fork of the code of the ICML 2022 paper [Generative Flow Networks for Discrete Probabilistic Modeling](https://arxiv.org/abs/2202.01361) by Dinghuai Zhang, Nikolay Malkin, Zhen Liu, Alexandra Volokhova, Aaron Courville, and Yoshua Bengio.
- Execute `run_densities2d.sh` to run the synthetic density experiments with the parameters specified in the script.
- Execute `run_ising.sh` to run the Ising model experiments with the parameters specified in the script.

**Inspecting results:**
Results (model parameters, evaluation metrics, and sample plots of final model) are saved in `./multiruns/<hostname>/<experiment_name>/<experiment_id_string>/` in `params`, `eval`, and `plots` respectively.
