from app.protocol import (
    StartMessage,
    StopMessage,
    PartialTranscriptMessage,
    FinalTranscriptMessage,
    InfoMessage,
    ErrorMessage,
)


class TestProtocolMessages:
    def test_start_message_valid(self):
        msg = StartMessage(
            lang="ko", clientSessionId="test-123", platformHint="youtube"
        )
        assert msg.type == "start"
        assert msg.lang == "ko"

    def test_start_message_defaults(self):
        msg = StartMessage(clientSessionId="test-123")
        assert msg.type == "start"
        assert msg.lang == "auto"
        assert msg.targetLang == "auto"
        assert msg.platformHint == "unknown"

    def test_stop_message_valid(self):
        msg = StopMessage()
        assert msg.type == "stop"

    def test_partial_transcript_message(self):
        msg = PartialTranscriptMessage(text="안녕하세요")
        assert msg.type == "partial"
        assert msg.text == "안녕하세요"

    def test_final_transcript_message(self):
        msg = FinalTranscriptMessage(text="Hello world")
        assert msg.type == "final"
        assert msg.text == "Hello world"

    def test_info_message(self):
        msg = InfoMessage(latencyMs=150, secondsUsed=10.5)
        assert msg.type == "info"
        assert msg.latencyMs == 150
        assert msg.secondsUsed == 10.5

    def test_error_message(self):
        msg = ErrorMessage(message="Connection failed", code="CONN_ERROR")
        assert msg.type == "error"
        assert msg.message == "Connection failed"
        assert msg.code == "CONN_ERROR"
