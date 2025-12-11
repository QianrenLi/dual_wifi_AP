"""RL algorithm implementations."""

# Import all algorithm implementations to trigger registration
from .sac_rnn_belief_seq_dist_v8 import SACRNNBeliefSeqDistV8, SACRNNBeliefSeqDistV8_Config
from .sac_rnn_belief_seq_dist_v9 import SACRNNBeliefSeqDistV9, SACRNNBeliefSeqDistV9_Config
from .sac_rnn_belief_seq_dist_v10 import SACRNNBeliefSeqDistV10, SACRNNBeliefSeqDistV10_Config

# These will be populated as we migrate more algorithms
# from .sac import *
# from .sac_rnn import *
# from .sac_belief import *
# from .sac_distributional import *

__all__ = [
    "SACRNNBeliefSeqDistV8",
    "SACRNNBeliefSeqDistV8_Config",
    "SACRNNBeliefSeqDistV9",
    "SACRNNBeliefSeqDistV9_Config",
    "SACRNNBeliefSeqDistV10",
    "SACRNNBeliefSeqDistV10_Config",
]