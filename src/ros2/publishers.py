from __future__ import annotations


class TopicPublisher:
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.last_message = None
        self._ros_publisher = None

    def attach_ros(self, node, message_type) -> None:
        self._ros_publisher = node.create_publisher(message_type, self.topic, 10)

    def publish(self, message) -> None:
        self.last_message = message
        if self._ros_publisher is not None:
            self._ros_publisher.publish(message)
