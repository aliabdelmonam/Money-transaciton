class ChannelError(Exception):
    pass


class RateLimitError(ChannelError):
    pass


class MediaTooLargeError(ChannelError):
    pass


class InvalidWebhookError(ChannelError):
    pass


class AuthenticationError(ChannelError):
    pass