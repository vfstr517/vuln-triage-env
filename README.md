---
title: VulnTriageEnv
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
---

# VulnTriageEnv

An OpenEnv-compliant environment simulating a real-world DevSecOps workflow. 

## Description and Motivation
Security engineers are often overwhelmed by automated vulnerability scanner outputs. This environment tasks an AI agent with reviewing raw vulnerability alerts, correlating them with an asset inventory, and taking appropriate triage actions. It bridges the gap between toy RL environments and actual enterprise security operations.

## Action Space
The agent outputs a `TriageAction` (JSON):
* `alert_id`: The ID of the alert being triaged.
* `action_type`: Must be one of `MARK_FALSE_POSITIVE`, `ASSIGN_TICKET`, `EMERGENCY_PATCH`, or `REQUEST_ASSET_INFO`.

## Observation Space
The agent receives an `Observation` containing:
* `open_alerts`: A list of active `VulnerabilityAlert` objects.
* `asset_context`: A dictionary mapping IPs to system roles (can be expanded via agent actions).

## Setup Instructions
1. Clone the repository.
2. Ensure Docker is installed.
3. Run `docker build -t vuln-triage-env .`
4. Run `docker run -e OPENAI_API_KEY="your_key" vuln-triage-env`

## Baseline
Run `python baseline.py` to execute a default gpt-4o agent against Task 1.