''' Reward Models

The functions in this module are used to train reward models to 
evaluate the quality of the representations learned by the representation models.

This is a relatively simple model consisting of a fully connected layer that converts
a representation to a scalar reward.
 '''

from clea.reward_models.model_definitions import RewardLearner
from clea.reward_models.eval_utils import get_pids_for_training, get_train_test_dataloaders, train_single_epoch_task_embeds, eval_model, calc_reward, generate_embeddings_task_conditioned
