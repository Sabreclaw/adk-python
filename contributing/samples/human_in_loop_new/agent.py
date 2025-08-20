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

import logging
from typing import Any

from google.adk import Agent
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_configs import InputConfig
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from pydantic import BaseModel

logger = logging.getLogger('google_adk.' + __name__)


class ReimbursementApproval(BaseModel):
  approval: bool = False
  """The approval status."""
  amount: float = 0.0
  """The amount of the reimbursement."""


def reimburse(purpose: str, amount: float, tool_context: ToolContext) -> str:
  """Reimburse the amount of money to the employee."""

  logger.info('Reimburse request: %s, %s', purpose, amount)

  return {
      'status': 'ok',
      'amount': amount,
  }


def before_tool_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> None:
  """Before tool callback."""

  if tool.name == 'reimburse':
    input_config = tool_context.input_config
    if input_config is None or input_config.data_input is None:
      logger.info('reimbursement approval is required')
      tool_context.request_input(
          InputConfig(
              text_input='Please approve or reject the reimbursement.',
              data_input=ReimbursementApproval(approval=False),
          )
      )
      return {
          'status': 'Manager approval is required. Please approve or reject.'
      }

    else:
      logger.info(input_config.data_input)
      reimbursement_approval = ReimbursementApproval.model_validate(
          input_config.data_input
      )

      logger.info('reimbursement_approval: %s', reimbursement_approval)
      if reimbursement_approval.approval:
        logger.info('reimbursement approved')
        return

      elif reimbursement_approval.amount > 0:
        logger.info(
            'reimbursement partially approved with new amount: %s',
            reimbursement_approval.amount,
        )
        args['amount'] = reimbursement_approval.amount
        return

      else:
        logger.info('reimbursement rejected')
        return {'status': 'rejected'}


root_agent = Agent(
    model='gemini-2.5-flash',
    name='reimbursement_agent',
    instruction="""
      You are an agent whose job is to handle the reimbursement process for
      the employees. Call reimburse tool to reimburse the employee. The request can be either
      approval or rejection, or partial approval with a new amount. If the amount is less than the request amount, you need to call out that to remind user.
""",
    tools=[reimburse],
    generate_content_config=types.GenerateContentConfig(temperature=0.1),
    before_tool_callback=before_tool_callback,
)
