from demos.experiments.peg_insert_sim.config import TrainConfig as PegInsertSimTrainConfig
from demos.experiments.peg_insert_pointcloud_sim.config import TrainConfig as PegInsertPointcloudSimTrainConfig

CONFIG_MAPPING = {
    "peg_insert_sim": PegInsertSimTrainConfig,
    "peg_insert_pointcloud_sim": PegInsertPointcloudSimTrainConfig,
}
