from pydantic import BaseModel # type: ignore
from typing import List


class Titles(BaseModel):
    titles: List[str]
   