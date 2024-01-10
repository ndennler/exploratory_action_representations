""" Representation Models

Representation models are used to learn representations of the data 
that are useful for downstream tasks in preference learning.

"""

from clea.representation_models.pretrained import PretrainedEncoder, AEPretrainedLearner, VAEPretrainedLearner

from clea.representation_models.auditory import RawAudioEncoder, RawAudioAE, RawAudioVAE
from clea.representation_models.visual import RawImageEncoder, RawImageAE, RawImageVAE
from clea.representation_models.kinetic import RawSequenceEncoder, Seq2Seq, Seq2SeqVAE

from clea.representation_models.train_model_utils import train_single_epoch, train_single_epoch_with_task_embedding