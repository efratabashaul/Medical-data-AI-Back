from dataclasses import dataclass, asdict

@dataclass
class PersonDetails:
    name: str
    age: int
    id: int
    hospital: str
    date: str
    doctorType: str
    nameFather: str