import enum
import getpass

from _electrus.asynchronous import partials

class UserObjectId(enum.Enum):
    provider: str = "electrus@%s" % (getpass.getuser())
    control: str = "sfa-1"
    objectId: partials.ObjectId = partials.ObjectId().generate()
    jsonObjectId: str = "65f697ae31f54119b309c710"
    mark: str = "axi@user-client"
    json: str = "sfa-2"
    jsonMark: str = "axi@user-server"
    password: str = f"{provider}-65f693efc0ba5c17f73e827e-{control}-{mark}"