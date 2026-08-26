"""Conversation lifecycle and scope transition service."""
from __future__ import annotations
from dataclasses import dataclass,replace
import json
from typing import Sequence
from uuid import uuid4
from paper_read_agent.domain.models import Conversation,Message
from paper_read_agent.persistence.repositories import SQLiteDomainRepository,utc_now

@dataclass(frozen=True,slots=True)
class RestoredSession:
    conversation:Conversation; messages:tuple[Message,...]; scope_history:tuple[dict,...]

class SessionService:
    def __init__(self,repository:SQLiteDomainRepository): self.repository=repository
    def create(self,title:str,scope_mode:str,paper_ids:Sequence[str]=())->Conversation:
        conv=Conversation(str(uuid4()),title,scope_mode,tuple(dict.fromkeys(paper_ids)))
        saved=self.repository.create_conversation(conv); self._event(saved); return saved
    def restore(self,conversation_id:str)->RestoredSession:
        conv=self.repository.get_conversation(conversation_id)
        if conv is None: raise KeyError(conversation_id)
        with self.repository.database.connect() as c:
            history=tuple({"scope_mode":r["scope_mode"],"paper_ids":tuple(json.loads(r["paper_ids_json"])),
                           "created_at":r["created_at"]} for r in c.execute(
                           "SELECT * FROM conversation_scope_events WHERE conversation_id=? ORDER BY created_at,event_id",(conversation_id,)))
        return RestoredSession(conv,self.repository.list_messages(conversation_id),history)
    def change_scope(self,conversation_id:str,scope_mode:str,paper_ids:Sequence[str]=())->Conversation:
        current=self.repository.get_conversation(conversation_id)
        if current is None: raise KeyError(conversation_id)
        updated=self.repository.update_conversation(replace(current,scope_mode=scope_mode,
            selected_paper_ids=tuple(dict.fromkeys(paper_ids)))); self._event(updated); return updated
    def delete(self,conversation_id:str)->None: self.repository.delete_conversation(conversation_id)
    def _event(self,conv:Conversation)->None:
        with self.repository.database.transaction() as c:
            c.execute("INSERT INTO conversation_scope_events VALUES (?,?,?,?,?)",
                      (str(uuid4()),conv.conversation_id,conv.scope_mode,json.dumps(conv.selected_paper_ids),utc_now()))
