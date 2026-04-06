from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class VulnerabilityAlert(BaseModel):
    alert_id: str
    severity: Literal['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    description: str
    target_ip: str

class Observation(BaseModel):
    open_alerts: List[VulnerabilityAlert]
    asset_context: dict

class TriageAction(BaseModel):
    alert_id: str
    action_type: Literal['MARK_FALSE_POSITIVE', 'ASSIGN_TICKET', 'EMERGENCY_PATCH', 'REQUEST_ASSET_INFO']

class Reward(BaseModel):
    value: float = Field(ge=-1.0, le=1.0)
    reasoning: str