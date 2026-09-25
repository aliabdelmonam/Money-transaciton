from datetime import datetime
from pydantic import BaseModel
from channels.models.user import User



class IncomingMessage(BaseModel):
    channel:str
    conversation_id:str
    sender:User
    message_id: str
    media_url:str
    timestamp: datetime