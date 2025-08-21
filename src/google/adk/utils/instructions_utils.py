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

import re

from ..agents.readonly_context import ReadonlyContext
from ..sessions.state import State

__all__ = [
    'inject_session_state',
]


async def inject_session_state(
    template: str,
    readonly_context: ReadonlyContext,
) -> str:
  """Populates values in the instruction template, e.g. state, artifact, etc.

  This method is intended to be used in InstructionProvider based instruction
  and global_instruction which are called with readonly_context.

  e.g.
  ```
  ...
  from google.adk.utils.instructions_utils import inject_session_state

  async def build_instruction(
      readonly_context: ReadonlyContext,
  ) -> str:
    return await inject_session_state(
        'You can inject a state variable like {var_name} or an artifact '
        '{artifact.file_name} into the instruction template.',
        readonly_context,
    )

  agent = Agent(
      model="gemini-2.0-flash",
      name="agent",
      instruction=build_instruction,
  )
  ```

  Args:
    template: The instruction template.
    readonly_context: The read-only context

  Returns:
    The instruction template with values populated.
  """

  invocation_context = readonly_context._invocation_context

  async def _async_sub(pattern, repl_async_fn, string) -> str:
    result = []
    last_end = 0
    for match in re.finditer(pattern, string):
      result.append(string[last_end : match.start()])
      replacement = await repl_async_fn(match)
      result.append(replacement)
      last_end = match.end()
    result.append(string[last_end:])
    return ''.join(result)

  async def _replace_match(match) -> str:
    var_name = match.group().lstrip('{').rstrip('}').strip()
    optional = False
    if var_name.endswith('?'):
      optional = True
      var_name = var_name.removesuffix('?')
    if var_name.startswith('artifact.'):
      var_name = var_name.removeprefix('artifact.')
      if invocation_context.artifact_service is None:
        raise ValueError('Artifact service is not initialized.')
      artifact = await invocation_context.artifact_service.load_artifact(
          app_name=invocation_context.session.app_name,
          user_id=invocation_context.session.user_id,
          session_id=invocation_context.session.id,
          filename=var_name,
      )
      if not var_name:
        raise KeyError(f'Artifact {var_name} not found.')
      return str(artifact)
    else:
      if not _is_valid_state_name(var_name):
        return match.group()
      if var_name in invocation_context.session.state:
        return str(invocation_context.session.state[var_name])
      else:
        if optional:
          return ''
        else:
          raise KeyError(f'Context variable not found: `{var_name}`.')

  # ESCAPE MECHANISM: Double-brace escaping for literal braces in instructions
  #
  # This implements a common templating escape pattern where double braces
  # become literal single braces, allowing users to include JSON, code, or
  # other brace-containing content in instruction templates.
  #
  # Processing order (CRITICAL - order matters!):
  # 1. First, escape double braces to avoid template processing
  # 2. Then, process single braces as template variables
  # 3. Finally, restore escaped braces as literals
  #
  # Example transformations:
  #   Input:    'Use {name} with config {{"type": "llm"}}'
  #   Step 1:   'Use {name} with config PLACEHOLDER'  (escape {{...}})
  #   Step 2:   'Use Alice with config PLACEHOLDER'   (process {name})
  #   Step 3:   'Use Alice with config {"type": "llm"}'  (restore literal)
  #
  # Why placeholders are needed:
  # - Direct replacement could interfere with regex pattern matching
  # - Placeholders ensure clean separation of escaping vs variable processing
  # - Unique strings prevent accidental collisions with user content

  # Step 1: Replace double braces with unique placeholders
  escaped_open = '__ADK_ESCAPED_OPEN_BRACE_PLACEHOLDER__'
  escaped_close = '__ADK_ESCAPED_CLOSE_BRACE_PLACEHOLDER__'
  template = template.replace('{{', escaped_open)
  template = template.replace('}}', escaped_close)

  # Step 2: Process single braces for template variables using regex pattern
  # Pattern r'{+[^{}]*}+' matches {variable_name} but not placeholders
  result = await _async_sub(r'{+[^{}]*}+', _replace_match, template)

  # Step 3: Restore escaped braces as literal braces in final output
  result = result.replace(escaped_open, '{')
  result = result.replace(escaped_close, '}')

  return result


def _is_valid_state_name(var_name):
  """Checks if the variable name is a valid state name.

  Valid state is either:
    - Valid identifier
    - <Valid prefix>:<Valid identifier>
  All the others will just return as it is.

  Args:
    var_name: The variable name to check.

  Returns:
    True if the variable name is a valid state name, False otherwise.
  """
  parts = var_name.split(':')
  if len(parts) == 1:
    return var_name.isidentifier()

  if len(parts) == 2:
    prefixes = [State.APP_PREFIX, State.USER_PREFIX, State.TEMP_PREFIX]
    if (parts[0] + ':') in prefixes:
      return parts[1].isidentifier()
  return False
