# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Annotated
from typing import Any
from typing import get_args
from typing import Union

from pydantic import Discriminator
from pydantic import RootModel
from pydantic import Tag

from ..utils.feature_decorator import experimental
from .base_agent_config import BaseAgentConfig
from .llm_agent_config import LlmAgentConfig
from .loop_agent_config import LoopAgentConfig
from .parallel_agent_config import ParallelAgentConfig
from .sequential_agent_config import SequentialAgentConfig


def _get_agent_class_values_for_config(config_class) -> set[str]:
  """Get all allowed agent_class values from a config class."""
  if agent_class_field := config_class.model_fields.get("agent_class"):
    return set(filter(bool, get_args(agent_class_field.annotation)))
  return set()


def _build_agent_class_mapping() -> dict[str, str]:
  """Build a mapping from agent_class values to tag names."""
  config_mappings = [
      (LlmAgentConfig, "LlmAgent"),
      (LoopAgentConfig, "LoopAgent"),
      (ParallelAgentConfig, "ParallelAgent"),
      (SequentialAgentConfig, "SequentialAgent"),
  ]

  return {
      value: tag_name
      for config_class, tag_name in config_mappings
      for value in _get_agent_class_values_for_config(config_class)
  }


# Cache the agent class mapping
_AGENT_CLASS_MAPPING = _build_agent_class_mapping()


def agent_config_discriminator(v: Any) -> str:
  """Discriminator function that returns the tag name for Pydantic."""
  if isinstance(v, dict):
    agent_class = v.get("agent_class", "LlmAgent")

    # Handle empty string case - default to LlmAgent
    if not agent_class:
      return "LlmAgent"

    # Look up the agent_class in our dynamically built mapping
    if agent_class in _AGENT_CLASS_MAPPING:
      return _AGENT_CLASS_MAPPING[agent_class]

    # For unknown agent classes, return "BaseAgent" to use BaseAgentConfig
    return "BaseAgent"

  raise ValueError(f"Invalid agent config: {v}")


# A discriminated union of all possible agent configurations.
ConfigsUnion = Annotated[
    Union[
        Annotated[LlmAgentConfig, Tag("LlmAgent")],
        Annotated[LoopAgentConfig, Tag("LoopAgent")],
        Annotated[ParallelAgentConfig, Tag("ParallelAgent")],
        Annotated[SequentialAgentConfig, Tag("SequentialAgent")],
        Annotated[BaseAgentConfig, Tag("BaseAgent")],
    ],
    Discriminator(agent_config_discriminator),
]


# Use a RootModel to represent the agent directly at the top level.
# The `discriminator` is applied to the union within the RootModel.
@experimental
class AgentConfig(RootModel[ConfigsUnion]):
  """The config for the YAML schema to create an agent."""
