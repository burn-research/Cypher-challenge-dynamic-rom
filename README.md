# CYPHER 2026: Hackathon on Reduced-Order Modeling for Advanced Combustion Systems
Challenge on reduced-order modeling for dynamical reacting flows organised within the context of the [CYPHER COST Action](https://cypher.ulb.be/)
<p align="center">
  <img src="https://github.com/burn-research/Cypher-challenge-dynamic-rom/blob/main/bundle/images/challenge_overview.png?raw=true" alt="Challenge Overview" width="90%">
</p>

## Overview
Combustion systems play a vital role in transportation, energy production, and residential heating. Nonetheless, current combustion systems heavily rely on fossil fuels, whose burning process is now acknowledged as a source of greenhouse gases and consequently responsible for climate change.
While completely dismantling our reliance on combustion is unfeasible for the so-called 'hard-to-abate industries' [[1](https://doi.org/10.1016/j.resconrec.2024.107796 )], existing combustion processes can significantly reduce their CO2 emissions. In this context, the CYPHER COST action is dedicated to advancing the understanding of Renewable Synthetic Fuels (RSFs) combustion, high-fidelity simulations, hybrid physics-based data-driven models, and self-updating digital twins. 

Reduced-order models are a powerful tool to develop digital replicas of physical systems, especially in the context of reacting flows, where numerical simulations are expensive and direct measurements are limited. Having access to a real-time digital model of a combustion system would allow us to control the combustion process much more precisely, increasing efficiency and reducing pollutant emissions. 

## Scope of the challenge
This challenge aims to establish a benchmark for reduced-order modeling of dynamical combustion systems. This first effort to provide a standard dataset and evaluation metric can be leveraged to foster consistency, facilitate model comparison, and accelerate progress in developing robust, generalizable reduced-order models for reactive flows. The public nature of the challenge, combined with its implementation on the Codabench https://doi.org/10.1016/j.patter.2022.100543 platform, ensures broad accessibility and promotes widespread dissemination within the research community.

Participants will be asked to submit a Python code that defines the reduced-order model, which will be trained after submission with the available data. All the models submitted will be evaluated on out-of-sample data. 

All the technical information can be found at https://cypher.ulb.be/data-challenge/

## Submission guidelines
Participants can join the challenge through the link https://www.codabench.org/competitions/18039/
The link will lead to the following page:

<p align="center">
  <img src="https://github.com/burn-research/Cypher-challenge-dynamic-rom/blob/main/bundle/images/main_page_codabench.png?raw=true" alt="homepage" width="90%">
</p>

After clicking on the highlighted "My Submissions" section, you will be able to submit your model:

<p align="center">
  <img src="https://github.com/burn-research/Cypher-challenge-dynamic-rom/blob/main/bundle/images/submission_2.png?raw=true" alt="homepage" width="90%">
</p>

An example file for submission is the 'sample_code_submission.zip' in the present GitHub folder. The [uncompressed folder](sample_code_submission) allows for exploring the structure of the Python code to be submitted. The only mandatory file that must be present at the moment of submission is the 'model.py' file. Every other kind of module is allowed in the submission folder, but additional data or pretrained models are not allowed in the present context. The organizers of the challenge reserve the right to check the submission files to verify that those restrictions are respected.

The model.py script must define a class, named 'model', an instance of which will be called from the ingestion program. The class can leverage tensorflow, pytorch, or scikit_learn to inherit, based on the user's preferences. The object must have 3 fundamental methods, which will be called when the file is submitted:
- ```preprocess(self, data_folder):``` takes as input the relative path to the [training data folder](bundle/input_data/train) and processes the data. The method must return an object (in the following referred to as D) that will be handled by the ```fit()``` method to train the model. The preprocessing can include every operation on the data in the folder that is useful to obtain better predictive capabilities.
- ```fit(self, D):``` trains the model based on the processed data contained in the object D. The method does not return outputs, but can update the object attributes (e.g., the weights of the neural networks).
- ```predict(self, valid_data_folder):``` takes as input the relative path to the [validation data folder](bundle/input_data/valid/). Another trivial but important consideration regards the data scaling, which should be handled in the same way both during the training and testing process. The scaling parameters must not be updated during inference.
The present method must return the sub-filter turbulent diffusivity of the progress variable, alpha_t, that will be used to evaluate the model. More details on the model form can be found in the document on the [Cypher website](https://cypher.ulb.be/data-challenge/).

## After submission
After submitting the file, the platform will start processing the data. Loading the app may take a few minutes. After the backend is ready and the docker image is loaded, you should see an output similar to the one represented below. Successive submissions without refreshing the page should be faster than the initial one.

<p align="center">
  <img src="https://github.com/burn-research/Cypher-challenge-dynamic-rom/blob/main/bundle/images/run.png?raw=true" alt="after submission" width="90%">
</p>

The window in red will output the statements from the backend. The model submitted can contain print statements that will be shown as output, which can be useful for debugging purposes. After the ingestion and scoring programs are done, you can click on the green highlighted button to download the outputs of the training, and check the log files with the output and errors, if any.

When multiple submissions are presented, the best one (lower scoring) can manually be selected to be added to the leaderboard:
<p align="center">
  <img src="https://github.com/burn-research/Cypher-challenge-dynamic-rom/blob/main/bundle/images/add_leaderboard.png?raw=true" alt="leaderboard" width="90%">
</p>

After being added to the leaderboard, the scoring should be visible in the results section:
<p align="center">
  <img src="https://github.com/burn-research/Cypher-challenge-dynamic-rom/blob/main/bundle/images/leaderboard.png?raw=true" alt="results" width="90%">
</p>


## Organizing committee

<p align="center">
  <img src="https://github.com/burn-research/Cypher-challenge-dynamic-rom/blob/main/bundle/images/organizing.png?raw=true" alt="organizing committee" width="90%">
</p>


