from . import register_baseline

def outage_extract(current_stats: dict) -> list:
    ## Iterate through current_stats to extract outage rates
    outage_rates = []
    for key, value in current_stats.items():
        if 'outage_rate' in key:
            outage_rates.append(value)
        if isinstance(value, dict):
            outage_rates.extend(outage_extract(value))
    return outage_rates

@register_baseline
class HardDeadLine:
    def __init__(self, state_cfg, initial_cmd):
        self.step_init = 0.05
        self.step_size = 0.05
        self.state_cfg = state_cfg
        self.init_tput = 1.0
        self.min_tput = -0.9
        self.initial_cmd = initial_cmd

    def act(self, current_stats, **kwargs):
        outage_rate = outage_extract(current_stats)
        if sum(outage_rate) > 0:
            self.step_size = self.step_init
            self.init_tput = max(self.min_tput, self.init_tput - self.step_size)
        else:
            self.step_size = self.step_size / 2
            self.init_tput += self.step_size
        self.initial_cmd = [self.initial_cmd[0], self.init_tput]
        return self.initial_cmd
