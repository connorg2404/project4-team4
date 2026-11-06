from typing import Annotated

from pydantic import BaseModel, Field, SecretStr


# Just like in Java, this Class will MODEL a Monster and its data

# BUT we're using something called Pydantic for structure and data validation
# (Notice the use of Annotated and Field for constraints)

# Inherit from BaseModel to get the Pydantic functionality
class BookModel(BaseModel):
    # This field must be greate than 0
    id: Annotated[int, Field(gt=0)]
    title: Annotated[str, Field()]
    author: Annotated[str, Field()]
    genre: Annotated[str, Field()]