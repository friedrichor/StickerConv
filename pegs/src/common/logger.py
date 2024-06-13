import os
import json
import logging
from datetime import datetime
from typing import Optional
from omegaconf import OmegaConf, DictConfig

from . import dist_utils


def init_logger_from_config(config: DictConfig, date_time: Optional[str] = None):
    if date_time is None:
        date_time = datetime.now().strftime("%b%d-%H_%M_%S")
    os.makedirs(os.path.join(config.run.outputs_dir, "logs"), exist_ok=True)
    log_path = os.path.join(config.run.outputs_dir, "logs", "{}.txt".format(date_time))
    logging.basicConfig(
        level=logging.INFO if dist_utils.is_main_process() else logging.WARN,
        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s || %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )
    logging.info("\n" + json.dumps(OmegaConf.to_container(config), indent=4, sort_keys=True))
