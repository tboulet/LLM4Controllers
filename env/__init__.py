from typing import Dict, Type
from env.base_meta_env import BaseMetaEnv
from env.gridworld import GridworldMetaEnv
from env.minigrid import MinigridMetaEnv


env_name_to_MetaEnvClass: Dict[str, Type[BaseMetaEnv]] = {
    "Gridworld": GridworldMetaEnv,
    "Minigrid": MinigridMetaEnv,
}
