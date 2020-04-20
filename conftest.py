import pytest
import os
@pytest.fixture(scope='session')
def app(request):
    from text_spotting import Server
    server = Server()
    return server.app