from pydantic import BaseModel, Field
from typing import Literal, Union


class LeftRightResponse(BaseModel):
    """Response model for left_right spatial relationship questions."""

    reasoning: str = Field(
        ..., description="Detailed explanation of the spatial relationship"
    )
    normalized_answer: Literal["left", "right"] = Field(
        ..., description="Either 'left' or 'right'"
    )

    @property
    def freeform_answer(self) -> str:
        """Generate freeform answer from reasoning and normalized answer."""
        return f"{self.reasoning} The answer is: {self.normalized_answer}"


class DistanceResponse(BaseModel):
    """Response model for distance estimation questions."""

    reasoning: str = Field(
        ..., description="Detailed explanation of the distance calculation"
    )
    normalized_answer: Union[float, str] = Field(
        ..., description="Distance in meters as a number"
    )

    @property
    def freeform_answer(self) -> str:
        """Generate freeform answer from reasoning and normalized answer."""
        return f"{self.reasoning} The answer is: {self.normalized_answer} meters"


class CountResponse(BaseModel):
    """Response model for counting questions."""

    reasoning: str = Field(..., description="Detailed explanation of the count")
    normalized_answer: Union[int, str] = Field(
        ..., description="Count as integer or word"
    )

    @property
    def freeform_answer(self) -> str:
        """Generate freeform answer from reasoning and normalized answer."""
        return f"{self.reasoning} The answer is: {self.normalized_answer}"


class MCQResponse(BaseModel):
    """Response model for multiple choice questions."""

    reasoning: str = Field(..., description="Detailed explanation of the choice")
    normalized_answer: Union[int, str] = Field(..., description="Region index number")

    @property
    def freeform_answer(self) -> str:
        """Generate freeform answer from reasoning and normalized answer."""
        return f"{self.reasoning} The answer is: Region {self.normalized_answer}"


# Map category to response model
RESPONSE_MODEL_MAP = {
    "left_right": LeftRightResponse,
    "distance": DistanceResponse,
    "count": CountResponse,
    "mcq": MCQResponse,
}
