from pydantic import BaseModel
from typing import List, Literal, Optional
from datetime import datetime


class TitleOne(BaseModel):
    message: str
   