from typing import Annotated

from pydantic import BaseModel, Field, SecretStr


# Just like in Java, this Class will MODEL a Monster and its data

# BUT we're using something called Pydantic for structure and data validation
# (Notice the use of Annotated and Field for constraints)

# Inherit from BaseModel to get the Pydantic functionality
class BookModel(BaseModel):
    # This field must be greate than 0
    id: Annotated[int, Field(gt=0)]
    # This field must be between 8 and 30 characters long
    title: Annotated[str, Field(min_length=2, max_length=200)]
    author: Annotated[str, Field(min_length=1, max_length=30)]
    # This field will be hidden when this object is returned,
    # and can be optional ( = none)
    description: Annotated[str, Field(min_length=0, max_length=500)]