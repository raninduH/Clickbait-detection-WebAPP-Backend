from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from datetime import datetime


class TitleOne(BaseModel):
    message: str
   

class Title_with_content(BaseModel):
    title: str
    content: str

class Title_with_content_batch(BaseModel):
    data: List[Title_with_content]

