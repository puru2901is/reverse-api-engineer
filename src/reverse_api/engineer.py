"""Reverse engineering module using Claude Agent SDK."""

import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, Any

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    ResultMessage,
)

from .utils import get_scripts_dir, get_timestamp
from .tui import ClaudeUI
from .messages import MessageStore


class APIReverseEngineer:
    """Uses Claude to analyze HAR files and generate Python API scripts."""

    def __init__(
        self,
        run_id: str,
        har_path: Path,
        prompt: str,
        model: Optional[str] = None,
        additional_instructions: Optional[str] = None,
        output_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        self.run_id = run_id
        self.har_path = har_path
        self.prompt = prompt
        self.model = model
        self.additional_instructions = additional_instructions
        self.scripts_dir = get_scripts_dir(run_id, output_dir)
        self.ui = ClaudeUI(verbose=verbose)
        self.usage_metadata: Dict[str, Any] = {}
        self.message_store = MessageStore(run_id, output_dir)

    def _build_analysis_prompt(self) -> str:
        """Build the prompt for Claude to analyze the HAR file."""
        base_prompt = f"""Analyze the HAR file at {self.har_path} and reverse engineer the APIs captured.

Original user prompt: {self.prompt}

Your task:
1. Read and analyze the HAR file to understand the API calls made
2. Identify authentication patterns (cookies, tokens, headers)
3. Extract request/response patterns for each endpoint
4. Generate a clean, well-documented Python script that replicates these API calls

The Python script should:
- Use the `requests` library
- Include proper authentication handling
- Have functions for each distinct API endpoint
- Include type hints and docstrings
- Handle errors gracefully
- Be production-ready

Save the generated Python script to: {self.scripts_dir / 'api_client.py'}
Also create a brief README.md in the same folder explaining the APIs discovered.
Always test your implementation to ensure it works. If it doesn't try again if you think you can fix it. You can go up to 5 attempts.
Sometimes websites have bot detection and that kind of things so keep in mind.
If you see you can't achieve with requests, feel free to use playwright with the real user browser with CDP to bypass bot detection.
No matter which implementation you choose, always try to make it production ready and test it.
"""
        if self.additional_instructions:
            base_prompt += f"\n\nAdditional instructions:\n{self.additional_instructions}"
        
        return base_prompt

    async def analyze_and_generate(self) -> Optional[Dict[str, Any]]:
        """Run the reverse engineering analysis with Claude."""
        self.ui.header(self.run_id, self.prompt, self.model)
        self.ui.start_analysis()
        
        # Save the prompt to messages
        self.message_store.save_prompt(self._build_analysis_prompt())

        options = ClaudeAgentOptions(
            allowed_tools=["Read", "Write", "Bash", "Glob", "Grep", "WebSearch", "WebFetch"],
            permission_mode="acceptEdits",
            cwd=str(self.scripts_dir.parent.parent),  # Project root
            model=self.model,
        )

        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(self._build_analysis_prompt())

                # Process response and show progress with TUI
                async for message in client.receive_response():
                    # Check for usage metadata in message if applicable
                    # (Note: Current SDK might not expose it easily, but we prepare for it)
                    if hasattr(message, 'usage') and isinstance(getattr(message, 'usage'), dict):
                        self.usage_metadata.update(getattr(message, 'usage'))

                    if isinstance(message, AssistantMessage):
                        last_tool_name = None
                        for block in message.content:
                            if isinstance(block, ToolUseBlock):
                                last_tool_name = block.name
                                self.ui.tool_start(block.name, block.input)
                                self.message_store.save_tool_start(block.name, block.input)
                            elif isinstance(block, ToolResultBlock):
                                is_error = block.is_error if block.is_error else False
                                
                                # Extract output from ToolResultBlock
                                output = None
                                if hasattr(block, 'content'):
                                    output = block.content
                                elif hasattr(block, 'result'):
                                    output = block.result
                                elif hasattr(block, 'output'):
                                    output = block.output
                                
                                tool_name = last_tool_name or "Tool"
                                self.ui.tool_result(tool_name, is_error, output)
                                self.message_store.save_tool_result(tool_name, is_error, str(output) if output else None)
                            elif isinstance(block, TextBlock):
                                self.ui.thinking(block.text)
                                self.message_store.save_thinking(block.text)
                    
                    elif isinstance(message, ResultMessage):
                        if message.is_error:
                            self.ui.error(message.result or "Unknown error")
                            self.message_store.save_error(message.result or "Unknown error")
                            return None
                        else:
                            script_path = str(self.scripts_dir / 'api_client.py')
                            self.ui.success(script_path)
                            
                            # Calculate estimated cost if we have usage data
                            if self.usage_metadata:
                                input_tokens = self.usage_metadata.get("input_tokens", 0)
                                output_tokens = self.usage_metadata.get("output_tokens", 0)
                                cache_creation_tokens = self.usage_metadata.get("cache_creation_input_tokens", 0)
                                cache_read_tokens = self.usage_metadata.get("cache_read_input_tokens", 0)
                                
                                # Claude Sonnet 4.5 pricing per million tokens:
                                # - Regular input: $3.00
                                # - Cache creation: $3.75
                                # - Cache read: $0.30
                                # - Output: $15.00
                                cost = (
                                    (input_tokens / 1_000_000 * 3.0) +
                                    (cache_creation_tokens / 1_000_000 * 3.75) +
                                    (cache_read_tokens / 1_000_000 * 0.30) +
                                    (output_tokens / 1_000_000 * 15.0)
                                )
                                self.usage_metadata["estimated_cost_usd"] = cost
                                
                                # Display usage breakdown
                                self.ui.console.print(f"  [dim]Usage:[/dim]")
                                if input_tokens > 0:
                                    self.ui.console.print(f"  [dim]  input: {input_tokens:,} tokens[/dim]")
                                if cache_creation_tokens > 0:
                                    self.ui.console.print(f"  [dim]  cache creation: {cache_creation_tokens:,} tokens[/dim]")
                                if cache_read_tokens > 0:
                                    self.ui.console.print(f"  [dim]  cache read: {cache_read_tokens:,} tokens[/dim]")
                                if output_tokens > 0:
                                    self.ui.console.print(f"  [dim]  output: {output_tokens:,} tokens[/dim]")
                                self.ui.console.print(f"  [dim]  total cost: ${cost:.4f}[/dim]")

                            result: Dict[str, Any] = {
                                "script_path": script_path,
                                "usage": self.usage_metadata
                            }
                            self.message_store.save_result(result)
                            return result

        except Exception as e:
            self.ui.error(str(e))
            self.message_store.save_error(str(e))
            self.ui.console.print(
                "\n[dim]Make sure Claude Code CLI is installed: "
                "npm install -g @anthropic-ai/claude-code[/dim]"
            )
            return None

        return None


def run_reverse_engineering(
    run_id: str,
    har_path: Path,
    prompt: str,
    model: Optional[str] = None,
    additional_instructions: Optional[str] = None,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """Synchronous wrapper for reverse engineering."""
    engineer = APIReverseEngineer(
        run_id=run_id,
        har_path=har_path,
        prompt=prompt,
        model=model,
        additional_instructions=additional_instructions,
        output_dir=output_dir,
        verbose=verbose,
    )
    return asyncio.run(engineer.analyze_and_generate())
