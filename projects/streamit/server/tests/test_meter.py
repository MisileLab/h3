from app.meter import UsageMeter


class TestUsageMeter:
    def test_add_bytes(self):
        UsageMeter._daily_usage.clear()

        seconds = UsageMeter.add_bytes("test-token", 48000)
        assert seconds == 1.0

    def test_add_multiple_bytes(self):
        UsageMeter._daily_usage.clear()
        UsageMeter.add_bytes("token1", 48000)
        UsageMeter.add_bytes("token1", 48000)
        assert UsageMeter.get_usage("token1") == 2.0

    def test_separate_tokens(self):
        UsageMeter._daily_usage.clear()
        UsageMeter.add_bytes("token1", 48000)
        UsageMeter.add_bytes("token2", 96000)
        assert UsageMeter.get_usage("token1") == 1.0
        assert UsageMeter.get_usage("token2") == 2.0

    def test_check_limit_within(self):
        UsageMeter._daily_usage.clear()
        UsageMeter.add_bytes("test-token", 24000)
        assert UsageMeter.check_limit("test-token", 3600) is True

    def test_check_limit_exceeded(self):
        UsageMeter._daily_usage.clear()

        UsageMeter.add_bytes("test-token", 3600 * 48000)
        assert UsageMeter.check_limit("test-token", 3600) is False

    def test_check_limit_no_limit(self):
        UsageMeter._daily_usage.clear()
        UsageMeter.add_bytes("test-token", 999999999)
        assert UsageMeter.check_limit("test-token", None) is True
