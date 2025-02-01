from typing import Dict, Type
from env.base_meta_env import BaseMetaEnv
from env.gridworld import Gridworld


env_name_to_MetaEnvClass: Dict[str, Type[BaseMetaEnv]] = {
    "Gridworld": Gridworld,
}
