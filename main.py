import asyncio
import logging
import ssl

from aiosmtpd.controller import Controller
from aiohttp import web

from api import routes
from smtp import SMTPHandler

BIND_HOST = "127.0.0.1"
BIND_PORT = 8025

STARTTLS_CERT_FILEPATH = 'cert.pem'
STARTTLS_KEY_FILEPATH = 'key.pem'

def main():
    logging.getLogger("asyncio").setLevel(logging.DEBUG)

    loop = asyncio.get_event_loop()
    loop.set_debug(True)

    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(STARTTLS_CERT_FILEPATH, STARTTLS_KEY_FILEPATH)

    controller = Controller(
        SMTPHandler(),
        hostname=BIND_HOST,
        port=BIND_PORT,
        loop=loop,
        server_hostname=BIND_HOST,
        require_starttls=False,
        tls_context=context
    )

    app = web.Application()
    app.add_routes(routes)

    controller.start()
    web.run_app(app)
    controller.stop()


if __name__ == '__main__':
    main()
