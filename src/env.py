import copy
from typing import Tuple
from .models import Observation, TriageAction, Reward, VulnerabilityAlert

class VulnTriageEnv:
    def __init__(self, task_level: str = "easy"):
        self.task_level = task_level.lower()
        self.internal_state = {}
        self.max_steps = 15
        self.current_step = 0

    def reset(self) -> Observation:
        self.current_step = 0
        
        if self.task_level == "easy":
            # Task 1: Find the false positives
            self.internal_state = {
                "alerts": [
                    VulnerabilityAlert(alert_id="A1", severity="CRITICAL", description="Windows IIS Exploit", target_ip="10.0.0.1"),
                    VulnerabilityAlert(alert_id="A2", severity="HIGH", description="Log4j", target_ip="10.0.0.2"),
                    VulnerabilityAlert(alert_id="A3", severity="MEDIUM", description="Nginx Path Traversal", target_ip="10.0.0.3")
                ],
                "assets": {
                    "10.0.0.1": {"os": "Linux Ubuntu", "role": "Web Server"}, 
                    "10.0.0.2": {"os": "Linux Debian", "role": "Java Backend"}, 
                    "10.0.0.3": {"os": "Windows Server", "role": "Active Directory"} 
                },
                "score": 0.0
            }
            
        elif self.task_level == "medium":
            # Task 2: Production vs Dev Prioritization
            self.internal_state = {
                "alerts": [
                    VulnerabilityAlert(alert_id="M1", severity="CRITICAL", description="SQL Injection", target_ip="10.1.0.50"),
                    VulnerabilityAlert(alert_id="M2", severity="HIGH", description="Outdated Redis", target_ip="10.1.0.51")
                ],
                "assets": {
                    "10.1.0.50": {"environment": "Production", "role": "Auth Database"},
                    "10.1.0.51": {"environment": "Development", "role": "Cache Server"}
                },
                "score": 0.0
            }

        elif self.task_level == "hard":
            # Task 3: Multi-step investigation (Requires gathering info first)
            self.internal_state = {
                "alerts": [
                    VulnerabilityAlert(alert_id="H1", severity="CRITICAL", description="RCE Vulnerability", target_ip="203.0.113.15")
                ],
                "assets": {}, # Empty! Agent must query this.
                "hidden_assets": { # Ground truth hidden from agent initially
                    "203.0.113.15": {"environment": "Production", "exposure": "Public Internet"}
                },
                "score": 0.0
            }
        
        return self.state()["observation"]

    def step(self, action: TriageAction) -> Tuple[Observation, Reward, bool, dict]:
        self.current_step += 1
        reward_val = 0.0
        reasoning = ""

        target_alert = next((a for a in self.internal_state["alerts"] if a.alert_id == action.alert_id), None)
        
        if not target_alert:
            return self.state()["observation"], Reward(value=-0.1, reasoning="Invalid alert_id"), False, {}

        asset_info = self.internal_state["assets"].get(target_alert.target_ip, {})

        # --- GRADING LOGIC ---
        
        # TASK 1 GRADER (Easy)
        if self.task_level == "easy":
            if action.action_type == "MARK_FALSE_POSITIVE":
                if ("Windows" in target_alert.description and asset_info.get("os") == "Linux Ubuntu") or \
                   ("Nginx" in target_alert.description and asset_info.get("os") == "Windows Server"):
                    reward_val = 0.33
                    reasoning = "Correctly identified false positive."
                else:
                    reward_val = -0.5
                    reasoning = "Incorrectly marked valid threat as false positive."
            elif action.action_type in ["ASSIGN_TICKET", "EMERGENCY_PATCH"]:
                if target_alert.alert_id == "A2":
                    reward_val = 0.34
                    reasoning = "Correctly triaged real vulnerability."
                else:
                    reward_val = -0.5
                    reasoning = "Wasted time on a false positive."

        # TASK 2 GRADER (Medium)
        elif self.task_level == "medium":
            is_prod = asset_info.get("environment") == "Production"
            if action.action_type == "EMERGENCY_PATCH" and is_prod:
                reward_val = 0.5
                reasoning = "Correctly emergency patched a production database."
            elif action.action_type == "ASSIGN_TICKET" and not is_prod:
                reward_val = 0.5
                reasoning = "Correctly assigned standard ticket for Dev server."
            else:
                reward_val = -0.5
                reasoning = "Failed to prioritize production OR caused unnecessary Dev downtime with emergency patch."

        # TASK 3 GRADER (Hard)
        elif self.task_level == "hard":
            if action.action_type == "REQUEST_ASSET_INFO":
                # Reveal the hidden info
                hidden_data = self.internal_state["hidden_assets"].get(target_alert.target_ip)
                if hidden_data:
                    self.internal_state["assets"][target_alert.target_ip] = hidden_data
                    reward_val = 0.2
                    reasoning = "Excellent: Gathered necessary context before deciding."
                else:
                    reward_val = 0.0
            elif action.action_type == "EMERGENCY_PATCH":
                if not asset_info: # They guessed without looking!
                    reward_val = -1.0
                    reasoning = "Catastrophic failure: Triaged critical alert blindly without requesting asset context."
                else:
                    reward_val = 0.8
                    reasoning = "Correctly patched internet-facing production server after verifying context."
            else:
                reward_val = -0.2
                reasoning = "Suboptimal action for this threat profile."

        # Remove processed alert (unless they just asked for info)
        if action.action_type != "REQUEST_ASSET_INFO":
            self.internal_state["alerts"] = [a for a in self.internal_state["alerts"] if a.alert_id != action.alert_id]

        # Clamp score between 0.0 and 1.0 (Rubric requirement)
        self.internal_state["score"] = max(0.0, min(1.0, self.internal_state["score"] + reward_val))
        done = len(self.internal_state["alerts"]) == 0 or self.current_step >= self.max_steps
        
        return self.state()["observation"], Reward(value=reward_val, reasoning=reasoning), done, {}

    def state(self) -> dict:
        raw_score = getattr(self, 'internal_score', getattr(self, 'score', 0.0))
        # Safely fetch the score, defaulting to 0.0 if it hasn't been set yet (like during reset)
        current_score = getattr(self, 'score', getattr(self, 'internal_score', 0.0))
        safe_score = max(0.01, min(0.99, current_score))
        return {
            "observation": Observation(
                open_alerts=self.internal_state.get("alerts", []),
                asset_context=self.internal_state.get("assets", {})
            ),
            "internal_score": self.internal_state.get("score", 0.0)
        }