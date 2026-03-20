from typing import List, Optional
from pydantic import BaseModel, Field

class Medication(BaseModel):
    name: str = Field(..., description="Name of the medication")
    dosage: Optional[str] = Field(None, description="Dosage information")

class MedicalExtraction(BaseModel):
    patient_name: Optional[str] = Field(None, description="Name of the patient")
    blood_type: Optional[str] = Field(None, description="Blood type of the patient")
    allergies: List[str] = Field(default_factory=list, description="List of known allergies")
    medications: List[Medication] = Field(default_factory=list, description="List of current medications")
    conditions: List[str] = Field(default_factory=list, description="List of medical conditions")
    emergency_actions: List[str] = Field(default_factory=list, description="Immediate life-saving actions to take")
    summary: Optional[str] = Field(None, description="Concise summary of the medical situation")
    is_verified: bool = Field(default=False, description="Whether the information has been verified via grounding")
