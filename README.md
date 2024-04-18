# exploratory_action_representations
A method for learning representations for preference learning from exploratory actions


# Run TNSE Plots
Allows users to examine relationship between user embeddings choices
'''
python get_tnse_graphs.py {signal type} {modality} {data_type} {file_name}
'''
## Setup For Project
'''
pip install requirements.txt // install requirements
pip install clea // install the requirements needed for clea module
'''


## Structure of Experiments
We conducted experiments manipulating the following variables:
- modality: {auditory, visual, kinetic}
- task_dependency: {independent by task (independent), conditioned on task (taskconditioned)}
- pretraining: {pretrained representations (named what the name of the pretrained model is, e.g., CLIP, xCLIP, etc.), directly from raw stimulus (raw)} 
- embedding_type: {Random, AE, VAE, Contrastive, Contrastive+AE} 
- signal : {has_item, searching, has_info, idle} or {all_signals} if task_dependency is taskconditioned

The naming convention for these models are:
```
{modality}&{task_dependency}&{pretraining}&{embedding_type}&{signal}&{representation_size}.pth
# if the task_dependency == taskconditioned, there is an additional network for conditioning saved with the name
{modality}&{task_dependency}&{pretraining}&{embedding_type}&{signal}&{representation_size}&embedder.pth
```

## Collecting Data
To collect data, first start an http server in the top level folder:

```
cd path/to/exploratory_action_representations
python3 -m http.server
```

create the experiment by running "generate_experiments.py".

Edit the variables at the top of data_collection/generate_experiments.py to reflect the participants'
current condition (auditory, visual, or kinetic as well as idle, searching, has_item, has_info). 

```
cd path/to/exploratory_action_representations/data_collection
python3 generate_experiments.py
```

