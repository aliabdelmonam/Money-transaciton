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


class MessageSendError(ChannelError):
    """Sending a (possibly split) message stopped at a failed part.

    ``sent`` are the parts the provider already accepted, ``unsent`` the
    texts of the failed part and every part after it. The provider's
    error is ``error`` (also the ``__cause__``).
    """

    def __init__(self, error: ChannelError, sent: list, unsent: list[str]):
        total = len(sent) + len(unsent)
        super().__init__(f"{error} (part {len(sent) + 1} of {total})" if total > 1 else str(error))
        self.error = error
        self.sent = sent
        self.unsent = unsent