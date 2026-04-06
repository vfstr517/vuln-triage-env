import copy
from typing import Tuple
from .models import Observation, TriageAction, Reward, VulnerabilityAlert

class VulnTriageEnv:
    def __init__(self, task_level: str = "easy"):
        self.task_level = task_level
        self.internal_state = {}
        self.max_steps = 10
        self.current_step = 0

    def reset(self) -> Observation:
        self.current_step = 0
        
        # Task 1: Easy - 3 alerts, 2 are obvious false positives, 1 is real
        if self.task_level == "easy":
            self.internal_state = {
                "alerts": [
                    VulnerabilityAlert(alert_id="A1", severity="CRITICAL", description="Windows IIS Exploit", target_ip="10.0.0.1"),
                    VulnerabilityAlert(alert_id="A2", severity="HIGH", description="Log4j", target_ip="10.0.0.2"),
                    VulnerabilityAlert(alert_id="A3", severity="MEDIUM", description="Nginx Path Traversal", target_ip="10.0.0.3")
                ],
                "assets": {
                    "10.0.0.1": {"os": "Linux Ubuntu", "role": "Web Server"}, # IIS exploit on Linux = False Positive
                    "10.0.0.2": {"os": "Linux Debian", "role": "Java Backend"}, # Log4j on Java = Real
                    "10.0.0.3": {"os": "Windows Server", "role": "Active Directory"} # Nginx on AD = False Positive
                },
                "score": 0.0
            }
        
        return self.state()["observation"]

    def step(self, action: TriageAction) -> Tuple[Observation, Reward, bool, dict]:
        self.current_step += 1
        reward_val = 0.0
        reasoning = ""

        # Find the targeted alert
        target_alert = next((a for a in self.internal_state["alerts"] if a.alert_id == action.alert_id), None)
        
        if not target_alert:
            return self.state()["observation"], Reward(value=-0.1, reasoning="Invalid alert_id"), False, {}

        asset_info = self.internal_state["assets"].get(target_alert.target_ip, {})

        # GRADING LOGIC (Task 1)
        if action.action_type == "MARK_FALSE_POSITIVE":
            if "Windows" in target_alert.description and asset_info.get("os") == "Linux Ubuntu":
                reward_val = 0.33
                reasoning = "Correctly identified false positive (Windows exploit on Linux)."
            elif "Nginx" in target_alert.description and asset_info.get("os") == "Windows Server":
                reward_val = 0.33
                reasoning = "Correctly identified false positive (Nginx on Windows AD)."
            else:
                reward_val = -0.5
                reasoning = "Incorrectly marked a valid threat as a false positive!"

        elif action.action_type == "ASSIGN_TICKET" or action.action_type == "EMERGENCY_PATCH":
            if target_alert.alert_id == "A2": # The real Log4j threat
                reward_val = 0.34
                reasoning = "Correctly triaged a real vulnerability."
            else:
                reward_val = -0.5
                reasoning = "Wasted engineering time assigning a false positive."
        
        elif action.action_type == "REQUEST_ASSET_INFO":
            reward_val = 0.0
            reasoning = "Gathered context."

        # Remove the processed alert if it was triaged
        if action.action_type in ["MARK_FALSE_POSITIVE", "ASSIGN_TICKET", "EMERGENCY_PATCH"]:
            self.internal_state["alerts"] = [a for a in self.internal_state["alerts"] if a.alert_id != action.alert_id]

        self.internal_state["score"] += reward_val
        done = len(self.internal_state["alerts"]) == 0 or self.current_step >= self.max_steps
        
        return self.state()["observation"], Reward(value=reward_val, reasoning=reasoning), done, {}

    def state(self) -> dict:
        return {
            "observation": Observation(
                open_alerts=self.internal_state.get("alerts", []),
                asset_context=self.internal_state.get("assets", {})
            ),
            "internal_score": self.internal_state.get("score", 0.0)
        }