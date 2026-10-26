# Agents

The agents are Python modules coordinated by a deterministic orchestrator:
context, diagnosis, knowledge, memory, and voice modules call the five mock
platform services through shared clients. The Anthropic tool-use loop runs in
the voice and agent path when configured. Offline evaluation uses the
deterministic planner and does not imply learned improvement.
