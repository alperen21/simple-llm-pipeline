"""LangGraph ReAct agent for the library assistant.

``state`` holds messages plus accumulated tool usage. ``nodes`` bind MCP
tools to the chat model and execute tool calls. ``graph`` compiles the loop
and ships a demo ``main``. ``prompts`` is the system policy for tool use and
untrusted retrieval.
"""
