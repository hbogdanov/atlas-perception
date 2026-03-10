from __future__ import annotations


class TopicSubscriber:
    def __init__(self, topic: str) -> None:
        self.topic = topic

    def receive(self):
        return None
