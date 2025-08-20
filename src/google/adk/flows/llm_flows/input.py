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

import json
from typing import AsyncGenerator
from typing import TYPE_CHECKING

from typing_extensions import override

from . import functions
from ...agents.invocation_context import InvocationContext
from ...agents.readonly_context import ReadonlyContext
from ...events.event import Event
from ...models.llm_request import LlmRequest
from ...tools.tool_configs import InputConfig
from ...tools.tool_configs import InputToolArguments
from ._base_llm_processor import BaseLlmRequestProcessor
from .functions import REQUEST_INPUT_FUNCTION_CALL_NAME

if TYPE_CHECKING:
  from ...agents.llm_agent import LlmAgent


class _InputLlmRequestProcessor(BaseLlmRequestProcessor):
  """Handles input information to build the LLM request."""

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    from ...agents.llm_agent import LlmAgent

    agent = invocation_context.agent
    if not isinstance(agent, LlmAgent):
      return
    events = invocation_context.session.events
    if not events:
      return

    request_input_function_responses = (
        dict()
    )  # function call id to input tool arguments
    for k in range(len(events) - 1, -1, -1):
      event = events[k]
      # Find the first event authored by user
      if not event.author or event.author != 'user':
        continue
      responses = event.get_function_responses()
      if not responses:
        return

      for function_response in responses:
        if function_response.name != REQUEST_INPUT_FUNCTION_CALL_NAME:
          continue

        # Find the FunctionResponse event that contains the user provided input
        # config
        request_input_function_responses[function_response.id] = (
            # TODO: Remove actual ['response'], this is a temporary solution.
            # ADK web will send a request that is always encapted in a
            # 'response' key.
            InputConfig(
                data_input=json.loads(function_response.response['response'])
            )
        )
      break

    if not request_input_function_responses:
      return

    for i in range(len(events) - 2, -1, -1):
      event = events[i]
      # Find the system generated FunctionCall event requesting the input config
      function_calls = event.get_function_calls()
      if not function_calls:
        continue

      tools_to_resume = dict()  # Function call id to input configs

      for function_call in function_calls:
        if function_call.id not in request_input_function_responses.keys():
          continue
        args = InputToolArguments.model_validate(function_call.args)

        tools_to_resume[args.original_function_call_id] = (
            request_input_function_responses[function_call.id]
        )
      if not tools_to_resume:
        continue

      # Find the original function call event that needs the input config
      for j in range(i - 1, -1, -1):
        event = events[j]
        function_calls = event.get_function_calls()
        if not function_calls:
          continue

        if any([
            function_call.id in tools_to_resume
            for function_call in function_calls
        ]):
          if function_response_event := await functions.handle_function_calls_async(
              invocation_context,
              event,
              {
                  tool.name: tool
                  for tool in await agent.canonical_tools(
                      ReadonlyContext(invocation_context)
                  )
              },
              # There could be parallel function calls that require input
              # response would be a dict keyed by function call id
              tools_to_resume.keys(),
              tools_to_resume,
          ):
            yield function_response_event
          return
      return


request_processor = _InputLlmRequestProcessor()
